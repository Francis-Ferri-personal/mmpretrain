from mmpretrain.registry import DATASETS
from .voc import VOC  # Asegúrate de que el import esté bien apuntado
from .categories import FEMORAL_ARTERY_CATEGORIES
from mmengine import list_from_file


@DATASETS.register_module()
class FemoralArteryDataset(VOC):
    """Custom dataset for femoral artery multi-label classification using Pascal VOC format."""

    METAINFO = {'classes': FEMORAL_ARTERY_CATEGORIES}

    def load_data_list(self):
        """Load images and ground truth labels."""
        data_list = []
        img_ids = list_from_file(self.image_set_path)

        for img_id in img_ids:
            img_path = self.backend.join_path(self.img_prefix, f'{img_id}.png')

            labels, labels_difficult = None, None
            if self.ann_prefix is not None:
                labels, labels_difficult = self._get_labels_from_xml(img_id)

            info = dict(
                img_path=img_path,
                gt_label=labels,
                gt_label_difficult=labels_difficult
            )
            data_list.append(info)

        return data_list
