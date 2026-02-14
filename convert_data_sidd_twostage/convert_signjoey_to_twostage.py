#!/usr/bin/env python3
"""
Converter: SignJoey format (61 keypoints) → Two-Stage SLG format
Convierte de .pt.gz a archivos .gloss, .skels, .files
"""

import argparse
import gzip
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


class SignJoeyToTwoStageConverter:
    """
    Convierte formato SignJoey (61 joints × 3D) a formato Two-Stage SLG
    """
    
    def __init__(self, num_joints: int = 61):
        """
        Args:
            num_joints: Número de joints (61 para SignIDD)
        """
        self.num_joints = num_joints
        self.joint_dim = num_joints * 3  # 61 × 3 = 183
    
    def load_signjoey_data(self, pt_gz_file: Path) -> List[Dict]:
        """
        Carga archivo .pt.gz en formato SignJoey
        
        Returns:
            Lista de samples con estructura:
            {
                'name': str,
                'text': str,
                'gloss': str,
                'sign': tensor [N_frames, 183],
                'signer': str
            }
        """
        print(f"📂 Loading {pt_gz_file}...")
        
        with gzip.open(pt_gz_file, 'rb') as f:
            samples = joblib.load(f)
        
        print(f"✓ Loaded {len(samples)} samples")
        return samples
    
    def convert_to_twostage_format(
        self, 
        samples: List[Dict], 
        output_dir: Path,
        split: str = "train"
    ):
        """
        Convierte samples de SignJoey a formato Two-Stage
        
        Crea 3 archivos:
        - {split}.gloss: glosas (una línea por secuencia)
        - {split}.skels: keypoints aplanados (una línea por secuencia)
        - {split}.files: nombres de secuencias
        
        Args:
            samples: Lista de diccionarios SignJoey
            output_dir: Directorio de salida
            split: "train", "dev", o "test"
        """
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        gloss_path = output_dir / f"{split}.gloss"
        skels_path = output_dir / f"{split}.skels"
        files_path = output_dir / f"{split}.files"
        
        print(f"\n{'='*70}")
        print(f"Converting {split} split to Two-Stage format")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}")
        print(f"Number of samples: {len(samples)}")
        print(f"Joints per frame: {self.num_joints}")
        print(f"Dimensions per frame: {self.joint_dim}")
        print(f"{'='*70}\n")
        
        with open(gloss_path, 'w', encoding='utf-8') as gloss_file, \
             open(skels_path, 'w', encoding='utf-8') as skels_file, \
             open(files_path, 'w', encoding='utf-8') as files_file:
            
            for sample in tqdm(samples, desc=f"Writing {split}"):
                
                # 1. Escribir glosa
                gloss = sample['gloss'].strip()
                gloss_file.write(gloss + '\n')
                
                # 2. Procesar y escribir keypoints
                sign = sample['sign']  # Shape: [N_frames, 183]
                
                # Convertir a numpy si es tensor
                if hasattr(sign, 'numpy'):
                    sign_np = sign.numpy()
                else:
                    sign_np = np.array(sign)
                
                # Verificar dimensiones
                num_frames, dim = sign_np.shape
                assert dim == self.joint_dim, \
                    f"Expected {self.joint_dim} dims, got {dim} in {sample['name']}"
                
                # Aplanar: [N_frames, 183] → [N_frames * 183]
                sign_flat = sign_np.reshape(-1)
                
                # Convertir a string separado por espacios
                sign_str = ' '.join([f"{val:.6f}" for val in sign_flat])
                skels_file.write(sign_str + '\n')
                
                # 3. Escribir nombre de archivo
                files_file.write(sample['name'] + '\n')
        
        print(f"\n✓ Files created:")
        print(f"  - {gloss_path}")
        print(f"  - {skels_path}")
        print(f"  - {files_path}")
        
        # Estadísticas
        self._print_statistics(samples, split)
    
    def _print_statistics(self, samples: List[Dict], split: str):
        """Imprime estadísticas del dataset"""
        
        num_frames_list = [sample['sign'].shape[0] for sample in samples]
        gloss_lengths = [len(sample['gloss'].split()) for sample in samples]
        
        print(f"\n{'='*70}")
        print(f"Statistics for {split} split:")
        print(f"{'='*70}")
        print(f"Total sequences: {len(samples)}")
        print(f"\nFrames per sequence:")
        print(f"  Min:    {min(num_frames_list)}")
        print(f"  Max:    {max(num_frames_list)}")
        print(f"  Mean:   {np.mean(num_frames_list):.2f}")
        print(f"  Median: {np.median(num_frames_list):.0f}")
        print(f"\nGloss words per sequence:")
        print(f"  Min:    {min(gloss_lengths)}")
        print(f"  Max:    {max(gloss_lengths)}")
        print(f"  Mean:   {np.mean(gloss_lengths):.2f}")
        print(f"  Median: {np.median(gloss_lengths):.0f}")
        print(f"{'='*70}\n")
    
    def create_metadata_files(
        self,
        samples: List[Dict],
        output_dir: Path,
        split: str = "train"
    ):
        """
        Crea archivos de metadata opcionales
        
        Crea:
        - signers.txt: nombre_secuencia,signer_id
        - texts.txt: nombre_secuencia|texto_traducido
        """
        
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # signers.txt
        signers_path = metadata_dir / f"{split}_signers.txt"
        with open(signers_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                signer = sample.get('signer', 'unknown')
                f.write(f"{sample['name']},{signer}\n")
        
        print(f"✓ Created: {signers_path}")
        
        # texts.txt
        texts_path = metadata_dir / f"{split}_texts.txt"
        with open(texts_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                text = sample.get('text', '')
                f.write(f"{sample['name']}|{text}\n")
        
        print(f"✓ Created: {texts_path}")


def validate_output(output_dir: Path, split: str, num_joints: int = 61):
    """
    Valida que los archivos generados sean correctos
    """
    
    print(f"\n{'='*70}")
    print(f"Validating {split} output files...")
    print(f"{'='*70}")
    
    gloss_path = output_dir / f"{split}.gloss"
    skels_path = output_dir / f"{split}.skels"
    files_path = output_dir / f"{split}.files"
    
    # Verificar existencia
    assert gloss_path.exists(), f"❌ Missing {gloss_path}"
    assert skels_path.exists(), f"❌ Missing {skels_path}"
    assert files_path.exists(), f"❌ Missing {files_path}"
    
    # Leer archivos
    with open(gloss_path, 'r') as f:
        glosses = f.readlines()
    
    with open(skels_path, 'r') as f:
        skels = f.readlines()
    
    with open(files_path, 'r') as f:
        files = f.readlines()
    
    # Verificar mismo número de líneas
    num_gloss = len(glosses)
    num_skels = len(skels)
    num_files = len(files)
    
    assert num_gloss == num_skels == num_files, \
        f"❌ Different number of lines: gloss={num_gloss}, skels={num_skels}, files={num_files}"
    
    print(f"✓ All files have {num_gloss} lines")
    
    # Verificar formato de skels (primeras 5 líneas)
    joint_dim = num_joints * 3  # 61 × 3 = 183
    
    for i, skel_line in enumerate(skels[:5]):
        values = skel_line.strip().split()
        num_values = len(values)
        
        # Debe ser múltiplo de joint_dim
        assert num_values % joint_dim == 0, \
            f"❌ Line {i}: {num_values} values (must be multiple of {joint_dim})"
        
        num_frames = num_values // joint_dim
        print(f"  Sequence {i}: {num_frames} frames × {num_joints} joints = {num_values} values ✓")
    
    # Verificar glosas no vacías
    empty_gloss = [i for i, g in enumerate(glosses) if not g.strip()]
    if empty_gloss:
        print(f"⚠  Empty glosses at lines: {empty_gloss[:10]}")
    else:
        print(f"✓ No empty glosses")
    
    # Verificar nombres únicos
    file_names = [f.strip() for f in files]
    unique_names = set(file_names)
    
    if len(file_names) != len(unique_names):
        duplicates = len(file_names) - len(unique_names)
        print(f"⚠  {duplicates} duplicate sequence names found")
    else:
        print(f"✓ All sequence names are unique")
    
    print(f"{'='*70}")
    print(f"✓ Validation passed for {split}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert SignJoey format (.pt.gz) to Two-Stage SLG format (.gloss, .skels, .files)'
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing train.pt.gz, dev.pt.gz, test.pt.gz')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for Two-Stage format files')
    parser.add_argument('--num-joints', type=int, default=61,
                        help='Number of joints (default: 61 for SignIDD)')
    parser.add_argument('--create-metadata', action='store_true',
                        help='Create optional metadata files (signers.txt, texts.txt)')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate output files after conversion')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Crear conversor
    converter = SignJoeyToTwoStageConverter(num_joints=args.num_joints)
    
    # Procesar cada split
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        pt_gz_file = input_dir / f"{split}.pt.gz"
        
        if not pt_gz_file.exists():
            print(f"\n⚠ Skipping {split}: {pt_gz_file} not found")
            continue
        
        # Cargar datos
        samples = converter.load_signjoey_data(pt_gz_file)
        
        # Convertir a formato Two-Stage
        converter.convert_to_twostage_format(samples, output_dir, split)
        
        # Crear metadata (opcional)
        if args.create_metadata:
            converter.create_metadata_files(samples, output_dir, split)
        
        # Validar
        if args.validate:
            validate_output(output_dir, split, args.num_joints)
    
    print(f"\n{'='*70}")
    print(f"✓ CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    for split in splits:
        if (output_dir / f"{split}.gloss").exists():
            print(f"  - {split}.gloss")
            print(f"  - {split}.skels")
            print(f"  - {split}.files")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


"""
EJEMPLO DE USO:

python convert_signjoey_to_twostage.py \
    --input-dir /Users/angiesanchez/Documents/paper-ECCV2/convert_data_raw_sidd/data \
    --output-dir ./Data/german \
    --num-joints 61 \
    --create-metadata \
    --validate

Esto creará:
./Data/german/
├── train.gloss
├── train.skels
├── train.files
├── dev.gloss
├── dev.skels
├── dev.files
├── test.gloss
├── test.skels
├── test.files
└── metadata/
    ├── train_signers.txt
    ├── train_texts.txt
    ├── dev_signers.txt
    ├── dev_texts.txt
    ├── test_signers.txt
    └── test_texts.txt
"""