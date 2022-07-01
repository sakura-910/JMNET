model = dict(
    type='JMNET',
    backbone=dict(
        type='eca_resnet18',
        pretrained=True,
        k_size=[3, 5, 5, 5],
    ),
    neck=dict(
        type='SS_ASPM',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        type='Cluster_Head',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='BCE_DiceLoss',
            weight=1,
            # alpha=0.8,
            # beta=0.7,
            # smooth=1e-6,
            # gamma=2
        ),
        loss_kernel=dict(
            type='BCE_DiceLoss',
            weight=0.5,
            # alpha=0.8,
            # beta=0.7,
            # smooth=1e-6,
            # gamma=2
        ),
        loss_emb=dict(
            type='UnESPLoss_v1',
            feature_dim=4,  # 4->8
            loss_weight=0.25
        )
    )
)
data = dict(
    batch_size=16,
    train=dict(
        type='JMNET_CTW',
        split='train',
        is_transform=True,
        img_size=640,
        short_size=640,
        kernel_scale=0.7,
        read_type='cv2'
    ),
    test=dict(
        type='JMNET_CTW',
        split='test',
        short_size=640,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=1000,
    optimizer='Adam',
    pretrain=' '
)
test_cfg = dict(
    min_score=0.88,
    min_area=16,
    bbox_type='poly',
    result_path='outputs/submit_ctw/'
)
