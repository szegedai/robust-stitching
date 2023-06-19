from torchvision import transforms as T

transformer = {

    'cifar10': [
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32,4),
            T.ToTensor(),
        ],
        [
            T.ToTensor(),
        ]
    ],

    'cifar100' : [
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32,4),
            T.ToTensor(),
        ],
        [
            T.ToTensor(),
        ]
    ],

    'mnist' : [
        [
            T.Resize((32,32)),
            T.Pad(4, padding_mode='edge'),
            T.RandomAffine(5, scale=(0.9, 1.1), shear=5, fillcolor=0),
            T.CenterCrop(32),
            T.ToTensor(),
        ],
        [
            T.Resize((32,32)),
            T.ToTensor(),
        ]
    ],

    'fashion' : [
        [
            T.Resize((32,32)),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32,4),
            T.ToTensor(),
        ],
        [
            T.Resize((32,32)),
            T.ToTensor(),
        ]
    ],

    'celeba' : [
        [
            T.ToPILImage(),
            T.Resize((256,256)),
            T.RandomCrop(224),
            T.Pad(10, padding_mode='edge'),
            T.RandomRotation(10),
            T.CenterCrop(224),
            T.ToTensor(),
            lambda x: x * 255 - 117
        ],
        [
            T.ToPILImage(),
            T.Resize((256,256)),
            T.CenterCrop(224),
            T.ToTensor(),
            lambda x: x * 255 - 117
        ]
    ],

    'svhn': [
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32,4),
            T.ToTensor()
        ], 
        [
            T.ToTensor()
        ]
    ],

    'imagenet': [
        [
            T.Resize(224),
            T.RandomResizedCrop(224),
            T.RandomCrop(32,4),
            T.ToTensor()
        ], 
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ]
    ]
}