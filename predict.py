import os
import cv2
import roop.globals
import numpy as np
from typing import Any
from PIL import Image
import uuid
from cog import BasePredictor, Input, Path
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
from roop.face_analyser import get_one_face

class Predictor(BasePredictor):
    def setup(self):
        """Initialize roop globals and CUDA settings"""
        roop.globals.execution_providers = ["CUDAExecutionProvider"]
        roop.globals.headless = True
        roop.globals.keep_fps = True
        roop.globals.keep_audio = True
        roop.globals.keep_frames = False
        roop.globals.many_faces = False
        roop.globals.video_encoder = "libx264"
        roop.globals.video_quality = 18
        roop.globals.reference_face_position = 0
        roop.globals.similar_face_distance = 0.6
        roop.globals.max_memory = 60
        roop.globals.execution_threads = 50

    def predict(
        self,
        source_image: Path = Input(description="Source face image"),
        target_image: Path = Input(description="Target image to swap face onto"),
        face_enhancer: bool = Input(description="Apply face enhancement", default=False)
    ) -> Path:
        try:
            # Create session directory
            session_id = str(uuid.uuid4())
            session_dir = f"temp/{session_id}"
            os.makedirs(session_dir, exist_ok=True)

            # Set up paths
            source_path = os.path.join(session_dir, "input.jpg")
            target_path = os.path.join(session_dir, "target.jpg")
            output_path = os.path.join(session_dir, "output.jpg")

            # Save input images
            source_img = Image.open(str(source_image))
            source_img.save(source_path)
            target_img = Image.open(str(target_image))
            target_img.save(target_path)

            # Check for faces in source image
            source_face = get_one_face(cv2.imread(str(source_path)))
            if source_face is None:
                raise ValueError("No face detected in source image")

            # Check for faces in target image
            target_face = get_one_face(cv2.imread(str(target_path)))
            if target_face is None:
                raise ValueError("No face detected in target image")

            # Set up frame processors
            frame_processors = ["face_swapper", "face_enhancer"] if face_enhancer else ["face_swapper"]

            # Run pre-checks
            for frame_processor in get_frame_processors_modules(frame_processors):
                if not frame_processor.pre_check():
                    raise ValueError(f"Pre-check failed for {frame_processor}")

            # Set global variables
            normalized_output_path = normalize_output_path(source_path, target_path, output_path)
            roop.globals.source_path = source_path
            roop.globals.target_path = target_path
            roop.globals.output_path = normalized_output_path
            roop.globals.frame_processors = frame_processors

            # Process the face swap
            from roop.core import start
            start()

            return Path(normalized_output_path)

        except Exception as e:
            raise ValueError(f"Error processing images: {str(e)}")