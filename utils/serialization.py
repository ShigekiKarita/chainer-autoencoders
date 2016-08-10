from chainer import serializers


def load(args, model, optimizer):
    # Init/Resume
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_npz(args.initmodel, model)
    if args.resume:
        print('Load optimizer state from', args.resume)
        serializers.load_npz(args.resume, optimizer)


def save(name, args, model, optimizer):
    # Save the model and the optimizer
    root = "res/" + name + "/"
    print('save the model')
    serializers.save_npz(root + 'mlp.model', model)
    print('save the optimizer')
    serializers.save_npz(root + 'mlp.state', optimizer)
    model.to_cpu()
