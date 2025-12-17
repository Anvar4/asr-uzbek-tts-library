"""
Model 2: Wav2Vec2 Fine-tuning for Uzbek ASR
Facebook'ning pre-trained Wav2Vec2 modelini o'zbek tili uchun fine-tuning qilish
"""

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np


class UzbekWav2Vec2Trainer:
    """
    O'zbek tili uchun Wav2Vec2 fine-tuning
    """
    
    def __init__(
        self,
        model_name="facebook/wav2vec2-base",  # yoki wav2vec2-large-xlsr-53
        vocab_dict=None
    ):
        """
        Args:
            model_name: Pre-trained model checkpoint
            vocab_dict: O'zbek alifbosi dictionary
        """
        self.model_name = model_name
        
        # O'zbek alifbosi
        if vocab_dict is None:
            self.vocab_dict = self._create_uzbek_vocab()
        else:
            self.vocab_dict = vocab_dict
        
        # Tokenizer va Processor
        self._setup_processor()
        
        # Model
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            vocab_size=len(self.vocab_dict),
            pad_token_id=self.processor.tokenizer.pad_token_id,
            ctc_loss_reduction="mean"
        )
        
        # Freeze feature extractor (optional)
        self.model.freeze_feature_encoder()
    
    def _create_uzbek_vocab(self):
        """O'zbek alifbosi yaratish"""
        vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CTC]': 2,  # CTC blank token
            ' ': 3,
        }
        
        # O'zbek harflari
        uzbek_chars = [
            'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z',
            'o\'', 'g\'', 'sh', 'ch', 'ng'
        ]
        
        for i, char in enumerate(uzbek_chars, start=4):
            vocab[char] = i
        
        return vocab
    
    def _setup_processor(self):
        """Processor setup"""
        # Tokenizer
        tokenizer = Wav2Vec2CTCTokenizer(
            self.vocab_dict,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token=" "
        )
        
        # Feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        
        # Processor
        self.processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
    
    def prepare_dataset(self, batch):
        """Dataset preprocessing"""
        # Audio
        audio = batch["audio"]
        
        # Feature extraction
        batch["input_values"] = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        
        # Text tokenization
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["text"]).input_ids
        
        return batch
    
    def compute_metrics(self, pred):
        """WER calculation"""
        from jiwer import wer
        
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        
        # Decode predictions
        pred_str = self.processor.batch_decode(pred_ids)
        
        # Ground truth
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, group_tokens=False)
        
        # Calculate WER
        wer_score = wer(label_str, pred_str)
        
        return {"wer": wer_score}
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir="./wav2vec2-uzbek",
        num_epochs=30,
        batch_size=8,
        learning_rate=3e-4,
        warmup_steps=500
    ):
        """Training"""
        
        @dataclass
        class DataCollatorCTCWithPadding:
            """Data collator for CTC"""
            processor: Wav2Vec2Processor
            padding: Union[bool, str] = True
            
            def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                # Split inputs and labels
                input_features = [{"input_values": feature["input_values"]} for feature in features]
                label_features = [{"input_ids": feature["labels"]} for feature in features]
                
                # Pad inputs
                batch = self.processor.pad(
                    input_features,
                    padding=self.padding,
                    return_tensors="pt",
                )
                
                # Pad labels
                with self.processor.as_target_processor():
                    labels_batch = self.processor.pad(
                        label_features,
                        padding=self.padding,
                        return_tensors="pt",
                    )
                
                # Replace padding with -100 (ignore in loss)
                labels = labels_batch["input_ids"].masked_fill(
                    labels_batch.attention_mask.ne(1), -100
                )
                
                batch["labels"] = labels
                
                return batch
        
        # Data collator
        data_collator = DataCollatorCTCWithPadding(processor=self.processor)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            group_by_length=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            num_train_epochs=num_epochs,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_total_limit=2,
            push_to_hub=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Train
        trainer.train()
        
        return trainer
    
    def predict(self, audio_path):
        """Inference"""
        import soundfile as sf
        
        # Load audio
        speech, sample_rate = sf.read(audio_path)
        
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
        
        # Process
        input_values = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Decode
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]
        
        return transcription


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Initialize trainer
    trainer = UzbekWav2Vec2Trainer(
        model_name="facebook/wav2vec2-large-xlsr-53"  # Multilingual pre-trained
    )
    
    print(f"Vocabulary size: {len(trainer.vocab_dict)}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # Example: Load dataset
    # train_dataset = load_dataset("your_uzbek_dataset", split="train")
    # eval_dataset = load_dataset("your_uzbek_dataset", split="test")
    
    # # Preprocess
    # train_dataset = train_dataset.map(trainer.prepare_dataset, remove_columns=train_dataset.column_names)
    # eval_dataset = eval_dataset.map(trainer.prepare_dataset, remove_columns=eval_dataset.column_names)
    
    # # Train
    # trained_model = trainer.train(
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     output_dir="./wav2vec2-uzbek",
    #     num_epochs=30,
    #     batch_size=8
    # )
    
    # # Inference
    # text = trainer.predict("path/to/uzbek_audio.wav")
    # print(f"Transcription: {text}")


# ==========================================
# DATASET YARATISH UCHUN HELPER
# ==========================================
def create_dataset_from_audio_files(audio_dir, transcript_file, output_dir):
    """
    Audio fayllar va transkriptlardan Hugging Face dataset yaratish
    
    Args:
        audio_dir: Audio fayllar papkasi
        transcript_file: Transkriptlar fayli (CSV yoki TXT)
        output_dir: Dataset saqlanadigan joy
    """
    from datasets import Dataset, DatasetDict, Audio
    import pandas as pd
    import os
    
    # Load transcripts
    # Format: audio_filename,text
    df = pd.read_csv(transcript_file)
    
    # Add full audio paths
    df['audio'] = df['filename'].apply(lambda x: os.path.join(audio_dir, x))
    
    # Create dataset
    dataset = Dataset.from_pandas(df[['audio', 'text']])
    
    # Cast audio column
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Split train/test
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Save
    dataset.save_to_disk(output_dir)
    
    print(f"Dataset saved to {output_dir}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")
    
    return dataset
