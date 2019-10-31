from facenet_pytorch import MTCNN
from get_link_idol.face_detection.data_manager import create_cropped_face_dataset
from control import Config


def crop_face(cfg):
    mtcnn = MTCNN(
        image_size=cfg.image_size,
        margin=cfg.margin,
        min_face_size=cfg.min_face_size,
        thresholds=cfg.threshold,
        factor=cfg.factor,
        prewhiten=cfg.prewhiten,
        device=cfg.device)

    create_cropped_face_dataset(mtcnn,
                                cfg.batch_size,
                                cfg.num_workers,
                                cfg.pin_memory)

    del mtcnn


if __name__ == '__main__':
    cfg = Config()
    crop_face(cfg)
