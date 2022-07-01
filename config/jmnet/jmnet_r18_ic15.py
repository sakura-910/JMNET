model = dict(
    type='JMNET',    
    backbone=dict(
        type='eca_resnet18',  
        pretrained=False,     
        k_size=[3, 5, 5, 5],  
        # groups=1,
        # base_width=64,
    ),
    neck=dict(
        type='SS_ASPM',
        in_channels=(64, 128, 256, 512),  
        out_channels=128,
    ),
    detection_head=dict(
        type='Cluster_Head',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='BCE_DiceLoss',
            weight=1,
            #alpha=0.8,
            # beta=0.7,
            # smooth=1e-6,
            #gamma=2
        ),
        loss_kernel=dict(
            type='BCE_DiceLoss',
            weight=0.5,
            #alpha=0.8,
            # beta=0.7,
            # smooth=1e-6,
            #gamma=2
        ),
        loss_emb=dict(
            type='UnESPLoss_v1',
            feature_dim=4,   #4->8
            loss_weight=0.25
        )
    )
)
data = dict(
    batch_size=16,
    train=dict(
        type='JMNET_IC15',
        split='train',
        is_transform=True,
        img_size=768,
        short_size=768,
        kernel_scale=0.5,
        read_type='cv2'
    ),
    test=dict(
        type='JMNET_IC15',
        split='test',
        short_size=768,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=900,
    optimizer='Adam'
)
test_cfg = dict(
    min_score=0.85,
    min_area=16,
    bbox_type='rect',
    result_path='outputs/submit_ic15_v1.zip'
)
