#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from datasets import create_dataset_from_args, DatasetConfig, ProcessingConfig
from model import LandmarkPredictor, ModelConfig, create_model, get_model_info
from trainer import (
    TrainingConfig, ModelTrainer, get_device, 
    set_all_seeds, print_system_info, validate_config
)
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


class MainTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.device)
        
        # 시드 설정
        set_all_seeds(args.seed)
        
        # 설정 생성
        self.dataset_config = self._create_dataset_config()
        self.processing_config = self._create_processing_config()
        self.model_config = self._create_model_config()
        self.training_config = self._create_training_config()
        
        print("=" * 60)
        print("모델 학습 시작")
        print("=" * 60)
        
    def _create_dataset_config(self) -> DatasetConfig:
        return DatasetConfig(
            deepspeak_base_dir=self.args.deepspeak_data_dir,
            kodf_features_root_dir=self.args.kodf_data_dir,
            fakeavceleb_dir=self.args.fakeavceleb_dir,
            dfdc_dir=self.args.dfdc_dir,
            deepfaketimit_dir=self.args.deepfaketimit_dir,
            faceforensics_dir=self.args.faceforensics_dir,
            load_deepspeak_real=self.args.load_deepspeak_real,
            load_deepspeak_fake=self.args.load_deepspeak_fake,
            load_kodf_real=self.args.load_kodf_real,
            load_kodf_fake=self.args.load_kodf_fake,
            load_fakeavceleb=self.args.load_fakeavceleb,
            load_dfdc_real=self.args.load_dfdc_real,
            load_dfdc_fake=self.args.load_dfdc_fake,
            load_deepfaketimit_fake=self.args.load_deepfaketimit_fake,
            load_faceforensics_real=self.args.load_faceforensics_real,
            load_faceforensics_fake=self.args.load_faceforensics_fake,
        )
    
    def _create_processing_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            desired_num_frames=self.args.input_frames,
            video_size=(224, 224),
            landmark_dim=478 * 2
        )
    
    def _create_model_config(self) -> ModelConfig:
        return ModelConfig(
            input_frames=self.args.input_frames,
            video_feature_dim=self.args.video_feature_dim,
            recon_target_dim=self.args.recon_target_dim,
            dropout_rate=self.args.dropout_rate
        )
    
    def _create_training_config(self) -> TrainingConfig:
        return TrainingConfig(
            num_epochs=self.args.num_epochs,
            batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            recon_loss_weight=self.args.recon_loss_weight,
            num_workers=self.args.num_workers,
            device=str(self.device),
            seed=self.args.seed,
            save_best_model=True,
            model_save_dir=self.args.model_save_dir,
            early_stopping_patience=self.args.early_stopping_patience,
            log_every_n_batches=self.args.log_every_n_batches
        )
    
    def create_datasets(self):
        print("\n📂 데이터셋 로딩 중...")
        
        datasets = {}
        for partition in ['train', 'val', 'test']:
            print(f"\n--- {partition.upper()} 데이터셋 ---")
            
            try:
                # 리팩토링된 방식으로 데이터셋 생성
                dataset = create_dataset_from_args(self.args, partition)
                
                if len(dataset) == 0:
                    print(f"⚠️  {partition} 데이터셋이 비어있습니다.")
                    datasets[partition] = None
                else:
                    print(f"✅ {partition} 데이터셋: {len(dataset)} 샘플")
                    datasets[partition] = dataset
                    
                    # 데이터 전처리 (옵션)
                    if self.args.preload_data and dataset:
                        print(f"🔄 {partition} 데이터 전처리 중...")
                        dataset.preload_data(batch_size=100)
                        
            except Exception as e:
                print(f"❌ {partition} 데이터셋 로딩 실패: {e}")
                datasets[partition] = None
        
        return datasets['train'], datasets['val'], datasets['test']
    
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset):
        print("\n🔄 데이터로더 생성 중...")
        
        def create_loader(dataset, partition, shuffle=False, drop_last=False):
            if dataset is None or len(dataset) == 0:
                return None
                
            return DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=self.args.num_workers,
                pin_memory=True,
                persistent_workers=True if self.args.num_workers > 0 else False
            )
        
        train_loader = create_loader(train_dataset, 'train', shuffle=True, drop_last=True)
        val_loader = create_loader(val_dataset, 'val', shuffle=False, drop_last=False)
        test_loader = create_loader(test_dataset, 'test', shuffle=False, drop_last=False)
        
        if train_loader is None:
            raise ValueError("❌ 학습 데이터셋이 비어있습니다. 데이터 경로와 로딩 옵션을 확인하세요.")
        
        # 배치 수 출력
        batch_info = [f"train: {len(train_loader)}"]
        if val_loader:
            batch_info.append(f"val: {len(val_loader)}")
        if test_loader:
            batch_info.append(f"test: {len(test_loader)}")
        
        print(f"📊 DataLoader 배치 수 → {', '.join(batch_info)}")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self):
        """모델 생성"""
        print("\n🧠 모델 생성 중...")
        
        model = create_landmark_predictor(
            input_frames=self.model_config.input_frames,
            video_feature_dim=self.model_config.video_feature_dim,
            recon_target_dim=self.model_config.recon_target_dim,
            dropout_rate=self.model_config.dropout_rate
        ).to(self.device)
        
        print(get_model_info(model))
        return model
    
    def create_optimizer_and_criterion(self, model):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        
        print(f" 손실함수: BCEWithLogitsLoss")
        print(f" 옵티마이저: Adam (lr={self.args.learning_rate})")
        
        return optimizer, criterion
    
    def run_training(self):
        """전체 학습 파이프라인 실행"""
        try:
            # 시스템 정보 출력
            if self.args.verbose:
                print_system_info()
            
            # 설정 검증
            if not validate_config(self.training_config):
                raise ValueError("설정 검증 실패")
            
            # 1. 데이터셋 생성
            train_dataset, val_dataset, test_dataset = self.create_datasets()
            
            # 2. 데이터로더 생성
            train_loader, val_loader, test_loader = self.create_dataloaders(
                train_dataset, val_dataset, test_dataset
            )
            
            # 3. 모델 생성
            model = self.create_model()
            
            # 4. 옵티마이저 및 손실함수
            optimizer, criterion = self.create_optimizer_and_criterion(model)
            
            # 5. 트레이너 생성 및 학습
            print(f"\n🏃 학습 시작!")
            print(f"💾 모델 저장 경로: {self.args.model_save_dir}")
            
            trainer = ModelTrainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                config=self.training_config,
                device=self.device
            )
            
            trainer.train(train_loader, val_loader, test_loader)
            
            print("🎉 학습 완료!")
            if trainer.best_model_path:
                print(f" 최고 성능: {trainer.best_model_path}")
            
        except KeyboardInterrupt:
            print("\n⏹️  사용자에 의해 학습이 중단되었습니다.")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 학습 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="멀티모달 딥페이크 탐지 모델 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 학습 파라미터
    training_group = parser.add_argument_group('🏃 학습 설정')
    training_group.add_argument('--num_epochs', type=int, default=10, 
                               help='학습 에포크 수')
    training_group.add_argument('--batch_size', type=int, default=16,
                               help='배치 크기')
    training_group.add_argument('--learning_rate', type=float, default=1e-3,
                               help='학습률')
    training_group.add_argument('--recon_loss_weight', type=float, default=0.5,
                               help='재구성 손실 가중치')
    training_group.add_argument('--early_stopping_patience', type=int, default=None,
                               help='조기 종료 patience (None=비활성화)')
    training_group.add_argument('--log_every_n_batches', type=int, default=10,
                               help='로그 출력 주기 (배치 단위)')
    
    # 시스템 설정
    system_group = parser.add_argument_group('💻 시스템 설정')
    system_group.add_argument('--device', type=str, default='auto',
                             help='디바이스 (auto, cpu, cuda, cuda:0, etc.)')
    system_group.add_argument('--num_workers', type=int, default=4,
                             help='DataLoader 워커 수')
    system_group.add_argument('--num_workers_preload', type=int, default=8,
                             help='데이터 전처리 워커 수')
    system_group.add_argument('--seed', type=int, default=42,
                             help='랜덤 시드')
    system_group.add_argument('--verbose', action='store_true',
                             help='상세 출력 모드')
    
    # 데이터 설정
    data_group = parser.add_argument_group('📂 데이터 설정')
    data_group.add_argument('--preload_data', action='store_true',
                           help='데이터 사전 로딩 활성화')
    data_group.add_argument('--cache_dir', type=str, default="cache",
                           help='캐시 디렉토리')
    data_group.add_argument('--model_save_dir', type=str, default="saved_models",
                           help='모델 저장 디렉토리')
    
    # 모델 설정
    model_group = parser.add_argument_group('🧠 모델 설정')
    model_group.add_argument('--input_frames', type=int, default=60,
                            help='입력 비디오 프레임 수')
    model_group.add_argument('--video_feature_dim', type=int, default=256,
                            help='비디오 특징 차원')
    model_group.add_argument('--recon_target_dim', type=int, default=956,
                            help='랜드마크 재구성 목표 차원 (478*2)')
    model_group.add_argument('--dropout_rate', type=float, default=0.3,
                            help='드롭아웃 비율')
    
    # 데이터셋 경로
    path_group = parser.add_argument_group('📁 데이터셋 경로')
    path_group.add_argument('--deepspeak_data_dir', type=str,
                           default="/media/NAS/DATASET/deepspeak/",
                           help='DeepSpeak 데이터셋 경로')
    path_group.add_argument('--kodf_data_dir', type=str,
                           default="/media/NAS/DATASET/KoDF/features_mtcnn/",
                           help='KoDF 데이터셋 경로')
    path_group.add_argument('--fakeavceleb_dir', type=str,
                           default="/media/NAS/DATASET/FakeAVCeleb_v1.2/",
                           help='FakeAVCeleb 데이터셋 경로')
    path_group.add_argument('--dfdc_dir', type=str,
                           default="/media/NAS/DATASET/DFDC-Official/",
                           help='DFDC 데이터셋 경로')
    path_group.add_argument('--deepfaketimit_dir', type=str,
                           default="/media/NAS/DATASET/DeepfakeTIMIT/",
                           help='DeepfakeTIMIT 데이터셋 경로')
    path_group.add_argument('--faceforensics_dir', type=str,
                           default="/media/NAS/DATASET/FaceForensics/",
                           help='FaceForensics++ 데이터셋 경로')
    
    # 데이터셋 로딩 옵션 (간소화)
    dataset_group = parser.add_argument_group('📋 데이터셋 로딩 옵션')
    
    # DeepSpeak
    dataset_group.add_argument('--load_deepspeak_real', action='store_true',
                              help='DeepSpeak 실제 데이터 로드')
    dataset_group.add_argument('--load_deepspeak_fake', action='store_true',
                              help='DeepSpeak 가짜 데이터 로드')
    
    # KoDF
    dataset_group.add_argument('--load_kodf_real', action='store_true',
                              help='KoDF 실제 데이터 로드')
    dataset_group.add_argument('--load_kodf_fake', action='store_true',
                              help='KoDF 가짜 데이터 로드')
    
    # 기타 데이터셋
    dataset_group.add_argument('--load_fakeavceleb', action='store_true',
                              help='FakeAVCeleb 데이터 로드')
    dataset_group.add_argument('--load_dfdc_real', action='store_true',
                              help='DFDC 실제 데이터 로드')
    dataset_group.add_argument('--load_dfdc_fake', action='store_true',
                              help='DFDC 가짜 데이터 로드')
    dataset_group.add_argument('--load_deepfaketimit_fake', action='store_true',
                              help='DeepfakeTIMIT 가짜 데이터 로드')
    dataset_group.add_argument('--load_faceforensics_real', action='store_true',
                              help='FaceForensics++ 실제 데이터 로드')
    dataset_group.add_argument('--load_faceforensics_fake', action='store_true',
                              help='FaceForensics++ 가짜 데이터 로드')
    
    return parser


def main():
    """메인 함수"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 학습 실행
    trainer = MainTrainer(args)
    trainer.run_training()


if __name__ == "__main__":
    main()