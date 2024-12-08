import torch
import numpy as np
import os
import gc
import cv2
from scipy import interpolate
import torch.nn as nn
import matplotlib.pyplot as plt
from .receptive_field import receptive_fields
from .load import getLayers,show_layers,summary,get_sublayers
from .tools import get_parameters, get_remains, get_optional_layers,\
    backward, make_layers, crop_bbox, Standardization, acc_label

class MemeNet(nn.Module):
    __slots__ = 'func', 'model', 'use_gpu', \
                'optional_node', 'avaliable_layers', 'rf', 'base', 'fc', \
                'rec', 'monitor', 'hooks', 'dist_maps', 'memes', \
                'num_memes', 'classifier', 'length', 'layers', \
                'mode'
    def __init__(self, prototype, keywords=["fc","classifier"],
                 classifier_layers = None, device_ids = [],
                 func=print):
        #prototype:Prototype network of reference
        #fc/base:lists of parameters
        #device_ids:list of gpu used
        #[]:use cpu
        super(MemeNet, self).__init__()
        self.func = func

        if len(device_ids) == 0:
            self.model = prototype
            output = "Using cpu to train prototype and classifer."
        else:
            self.model = nn.DataParallel(prototype, device_ids = device_ids).cuda()
            output = "Using gpu " + ','.join(map(str, device_ids)) + " to train."
        self.func(output)
        self.use_gpu = device_ids

        if self.use_gpu:
            self.func("Due to the use of gpu for training, it is recommended to add module. before the name of the layer")

        self.demonstrate()

        self.optional_node, self.avaliable_layers = get_optional_layers(getLayers(self.model),keywords)
        self.rf = receptive_fields(self.model)

        self.keywords = keywords

        self.rec =False
        self.monitor = {}
        self.hooks = {}
        
        self.dist_maps = {}
        self.memes = {}
        self.num_memes = 0

        if classifier_layers is None:
            self.classifier = nn.Sequential()
            self.length = 0
        else:
            self.length = len(classifier_layers)
            self.classifier = classifier_layers
            if len(self.use_gpu):
                self.classifier.to(self.use_gpu[0])

        self.layers = {}
        self.mode = 'dist'

    def get_idx(self, name):
        real_name = self.get_name(name)
        if real_name in self.optional_node:
            return self.optional_node.index(real_name)
        else:
            return -1

    def warm_only(self):
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = False

    def start_up(self):
        base,fc = get_parameters(self.model,self.keywords)
        for param in get_remains(self.model,[base,fc]):
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in base:
            param.requires_grad = False
        for param in fc:
            param.requires_grad = False

    def warm_train(self, train_loader, valid_loader, lr = [5e-3, 1e-2],
                   warm_lr_step_size = 0, loss_func = torch.nn.CrossEntropyLoss(),
                   epochs=3, weight_decay = 0, mode=0, vis_iter = 5):
        self.func("Start training prototype...")
        self.warm_only()
        if isinstance(lr, list):
            lr_base, lr_fc = lr
        else:
            lr_base = lr
            lr_fc = lr
        self.func("The learning rate is {},{}(extractor,classifier). The lr_step is {}, the loss_func is {}."
                  "The weight_decay is {}.".format(lr_base,lr_fc,warm_lr_step_size,loss_func,weight_decay))


        base, fc = get_parameters(self.model, self.keywords)
        warm_optimizer_specs = \
                [{'params': base, 'lr': lr_base},
                 {'params': fc, 'lr': lr_fc},
                 ]
        
        if mode:
            warm_optimizer = torch.optim.Adam(warm_optimizer_specs, weight_decay = weight_decay)
        else:
            warm_optimizer = torch.optim.SGD(warm_optimizer_specs, weight_decay = weight_decay,  momentum = 0.9)
        if warm_lr_step_size != 0:
            warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(warm_optimizer,
                        step_size=warm_lr_step_size, gamma=0.1)

        count = 0
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                if len(self.use_gpu):
                    images, labels = images.cuda(), labels.cuda()
                warm_optimizer.zero_grad()
                outputs = self.model(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                warm_optimizer.step()
                if warm_lr_step_size != 0:
                    warm_lr_scheduler.step()
                count += 1
                if count % vis_iter == 0:
                    self.func("iter " + str(count) + "| loss: " + str(loss.item()))
            if epoch % (vis_iter * 5) == 0:
                self.func("epoch:" + str(epoch + 1) + "/" + str(epochs))
                accuracy = self.test_prototype(valid_loader)
                self.func("Acc: " + str(accuracy))
        self.func("final acc: " + str(accuracy))

    def warm_train_once(self, train_data, train_label,
                        loss_func = torch.nn.CrossEntropyLoss()):
        self.func("Start training prototype once...")
        self.warm_only()
        base, fc = get_parameters(self.model, self.keywords)
        warm_optimizer_specs = \
            [{'params': base, 'lr': 5e-3},
             {'params': fc, 'lr': 1e-2},
             ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
        if len(self.use_gpu):
            train_data, train_label = train_data.cuda(), train_label.cuda()
        loss = backward(train_data, train_label, self.model, warm_optimizer, loss_func)
        output = "Data amount:"+ len(train_data) + "| Loss: "+loss.item()
        self.func(output)

    def baseline_train(self, train_loader, valid_loader,
                       loss_func = torch.nn.CrossEntropyLoss(), epochs=3,
                       lr=1e-3, weight_decay = 0):
        self.func("Start training the prototype classifier...")
        self.warm_only()
        self.func("The learning rate is {}. The loss_func is {}."
                  "The weight_decay is {}.".format(lr, loss_func, weight_decay))
        _, fc = get_parameters(self.model, self.keywords)
        baseline_optimizer = torch.optim.Adam(fc, lr=lr, weight_decay = weight_decay)

        count = 0
        vis_iter = 5

        for epoch in range(epochs):
            self.model.train()
            self.func("epoch:" + str(epoch + 1) + "/" + str(epochs))
            for i, (images, labels) in enumerate(train_loader):
                if len(self.use_gpu):
                    images, labels = images.cuda(), labels.cuda()
                baseline_optimizer.zero_grad()
                outputs = self.model(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                baseline_optimizer.step()
                count += 1
                if count % vis_iter == 0:
                    self.func("iter " + str(count) + "| loss: " + str(loss.item()))
            baseline_optimizer.zero_grad()
            accuracy = self.test_prototype(valid_loader)
            self.func("Acc: " + str(accuracy))
        self.func("final acc: " + str(accuracy))

    def test_prototype(self, test_loader):
        self.func("Testing prototype...")
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                if len(self.use_gpu):
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            return accuracy

    def test_model(self, test_loader):
        self.func("Testing model...")
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                if self.use_gpu:
                    images = images.cuda()
                    labels = labels.to(self.use_gpu[0])
                features = self.get_features(images)
                outputs = self.classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            return accuracy

    def test_prototype_label(self, test_loader):
        self.func("Label by label classification...")
        return acc_label(self.model, test_loader,len(self.use_gpu))

    def test_model_label(self, test_loader):
        self.func("Label by label classification(main model)...")
        return acc_label(self, test_loader, len(self.use_gpu))

    def demonstrate(self,prototype=True):
        if prototype==True:
            self.func('The structure of the backbone is:')
            show_layers(self.model, func = self.func)
        else:
            assert len(self.classifier)!=0,"Classifier layers must be given"
            self.func('The structure of the classifier is:')
            show_layers(self.classifier, func = self.func)

    def visualize(self, keywords=["fc","classifier"]):
        self.func('The structure of the memenet is:')
        backbone_length_limit=20
        intermediate = 5
        intermediate_text = 10
        branch_length_limit = 21
        if len(self.memes) != 0:
            self.extend_summary(self.model, keywords=keywords, intermediate = intermediate,
               intermediate_text = intermediate_text, branch_length_limit = branch_length_limit, 
               backbone_length_limit = backbone_length_limit)
            tab = (backbone_length_limit + 2 * intermediate + intermediate_text) * " "
            if len(self.classifier) != 0:
                summary(self.classifier, self.hooks, backbone_length_limit = branch_length_limit, tab=tab, func = self.func)
            else:
                self.func(tab+"output".center(branch_length_limit))
        else:
            summary(self.model, self.hooks, backbone_length_limit = backbone_length_limit, func = self.func)

    def add_monitor(self, idx):
        name = self.optional_node[idx]
        assert name in self.optional_node, "Layer "+str(name)+" monitoring is not allowed, available monitoring nodes: "+",".join(self.optional_node.keys())
        self.func("Adding agent monitoring the output of layer "+str(name))
        if idx not in self.monitor:
            self.monitor[idx]=[]
            def forward_hook(module,input,output):
                if self.rec:
                    self.monitor[idx].append(torch.transpose(output[0],1,2).unsqueeze(-1))
            handle = self.avaliable_layers[idx].register_forward_hook(forward_hook)
            self.hooks[idx] = handle

    def switch_monitoring(self):
        if self.rec:
            self.rec = False
            output = "All nodes stop monitoring, current status: " + str(self.rec)
        else:
            self.rec = True
            output = "All nodes start monitoring, current status: " + str(self.rec)
        self.func(output)

    def del_hooks(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks={}
        self.monitor={}
        self.func("All monitor nodes deleted.")

    def add_memes(self, kernels, func, zscore_module = {}, used_target ={}):
        # layers: special layers made by .tools.make_layers
        # zscore_module: dict(optional)
        # Used for whitening (only for specific module)
        # used_target: dict(optional)
        # key: where(str), value: kernel size(list of tuples)
        # Specifies the meme to be added
        if len(used_target) == 0:
            used_kernels = kernels
        else:
            used_kernels = dict()
            for where in used_target:
                sizes = used_target[where]
                for size in sizes:
                    key = "{}_{}_{}".format(where,size[0],size[1])
                    assert key in kernels
                    used_kernels[key] = kernels[key]
        layers, mode = make_layers(used_kernels, func, zscore_module)
        self.mode = mode
        self.add_layers(layers)

    def add_layers(self,layers):
        for key in layers:
            where = int(key.split("_")[0])
            layer = layers[key]
            if key not in self.memes:
                self.memes[key]=dict()
            self.num_memes += layer.weight.shape[0]
            if self.use_gpu:
                layer = layer.cuda()
            handle = self.avaliable_layers[where].register_forward_hook(self.get_hook_func(layer, key))
            self.memes[key]=handle
        if not self.layers:
            self.layers=layers
        self.func(str(self.num_memes)+" memes considered.")

    def get_hook_func(self, layer, key):
        # Calculate similarity and downsample
        def func(module, input, output):
            ED = layer(torch.transpose(output[0],1,2).unsqueeze(-1))
            if key not in self.dist_maps:
                self.dist_maps[key] = []
            self.dist_maps[key].append(ED)
        return func

    def add_classifier(self,num_memes,out_features):
        assert len(self.classifier)==1,"Not suitable for complex classifier"
        layer = self.classifier[0]
        add_memes = num_memes - layer.in_features
        add_output = out_features - layer.out_features
        new_layer=nn.Linear(num_memes, out_features)
        temp_weight = torch.cat((layer.weight, \
                            torch.zeros([layer.out_features, add_memes])), 1)
        new_weight = torch.cat((temp_weight, \
                                torch.zeros([add_output, temp_weight.shape[1]])), 0)
        new_bias = torch.cat((layer.bias, torch.zeros([add_output])),0)
        new_layer.weight.detach().copy_(new_weight)
        new_layer.bias.detach().copy_(new_bias)
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', new_layer)
        self.demonstrate(False)

    def load_classifier(self,classifier):
        #classifier:Arbitrary model for output results
        self.classifier = classifier

    def making_classifier(self,num_hiddens,num_classes):
        assert isinstance(num_hiddens,tuple),"Params num_hiddens must be tuple."
        self.func("A "+str(len(num_hiddens)+1)+"-layers neural network is constructed. There are "+str(num_hiddens)+" neurons in the hidden layer, and there are "+str(num_classes)+" outputs in total.")
        self.classifier = nn.Sequential()
        if len(num_hiddens)!=0:
            for i in range(len(num_hiddens)):
                if i==0:
                    self.classifier.add_module('classifier_' + str(i + 1), nn.Linear(self.num_memes, num_hiddens[i]))
                else:
                    self.classifier.add_module('classifier_' + str(i + 1), nn.Linear(num_hiddens[i-1], num_hiddens[i]))
                self.classifier.add_module('relu_' + str(i + 1), nn.ReLU())
            self.classifier.add_module('classifier_' + str(len(num_hiddens)+1), nn.Linear(num_hiddens[i], num_classes))
        else:
            self.classifier.add_module('classifier', nn.Linear(self.num_memes, num_classes))
        self.demonstrate(False)
        self.length = len(self.classifier)
        if len(self.use_gpu):
            self.classifier.to(self.use_gpu[0])

    def del_memes(self):
        for meme_hook in self.memes.values():
            meme_hook.remove()
        self.memes={}
        self.num_memes=0
        self.classifier = nn.Sequential()
        self.func("All memes deleted.")

    def get_features(self, x):
        for key in self.dist_maps:
            self.dist_maps[key] = []
        with torch.no_grad():
            self.model.eval()
            #x is consistent
            self.model(x)
            #the result is consistent
            dist_maps = self.get_sim()
            features = torch.cat(dist_maps, 1)
        return features

    def get_features_with_mask(self, x, mask):
        for key in self.dist_maps:
            self.dist_maps[key] = []
        with torch.no_grad():
            self.model.eval()
            #x is consistent
            self.model(x)
            #the result is consistent
            dist_maps = self.get_sim_with_mask(mask)
            features = torch.cat(dist_maps, 1)
        return features

    def forward(self, x):
        assert self.num_memes != 0, "At least one meme must be given"
        features = self.get_features(x)
        if self.length == 0:
            return features
        else:
            return self.classifier(features)

    def gather_map(self,dist_map):
        if len(self.use_gpu) > 1:
            used_device = [item.device.index for item in dist_map]
            main_device = self.use_gpu[0]
            idx = [i[0] for i in sorted(enumerate(used_device), key=lambda x: x[1])]
            used_map = nn.parallel.gather([dist_map[i] for i in idx], self.use_gpu).to(main_device)
        else:
            used_map = torch.cat(dist_map,axis = 1)
        return used_map

    def get_sim(self, refresh = True):
        dist_maps = []
        for key in self.dist_maps.keys():
            dist_map = self.dist_maps[key]
            used_map = self.gather_map(dist_map)
            if self.mode == 'dist':
                ans = torch.min(torch.min(used_map, 2).values, 2).values
            else:
                ans = torch.max(torch.max(used_map, 2).values, 2).values
            dist_maps.append(ans)
            if refresh: self.dist_maps[key] = []
        return dist_maps

    def get_sim_with_mask(self, mask, refresh = True):
        dist_maps = []
        for key in self.dist_maps.keys():
            dist_map = self.dist_maps[key]
            used_map = self.gather_map(dist_map)
            if self.mode == 'dist':
                ans = torch.min(torch.min(used_map, 2).values, 2).values
            else:
                ans = torch.max(torch.max(used_map, 2).values, 2).values
            ans = torch.mul(ans, torch.tensor(mask[key]))
            dist_maps.append(ans)
            if refresh: self.dist_maps[key] = []
        return dist_maps

    def formal_train(self, train_loader, valid_loader, epochs=3, mode = 1,
                     loss_func = torch.nn.CrossEntropyLoss(),
                     lr=1e-3, formal_lr_step_size =500,
                     weight_decay=0, vis_iter = 5, test = True):
        self.func("Start formal training...")
        self.start_up()
        if mode==1:
            formal_optimizer = torch.optim.Adam(self.classifier.parameters(),lr=lr, weight_decay=weight_decay)
            self.func("Using adam optimizer, the loss function is {}, learning rate is {}, weight decay is {}.".format(loss_func,lr,weight_decay))
            formal_lr_scheduler = torch.optim.lr_scheduler.StepLR(formal_optimizer,
                        step_size=formal_lr_step_size, gamma=0.1)
        else:    
            formal_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            self.func("Using SGD optimizer, the loss function is {}, learning rate is {}, weight decay is {}.".format(loss_func,lr,weight_decay))
        count = 0
        for epoch in range(epochs):
            self.func("epoch:" + str(epoch + 1) + "/" + str(epochs))
            self.model.train()
            for i, (images, labels) in enumerate(train_loader):
                formal_optimizer.zero_grad()
                if self.use_gpu:
                    images = images.cuda()
                    labels = labels.to(self.use_gpu[0])
                #images is consistent
                features = self.get_features(images)
                #feature is consistent
                outputs = self.classifier(features)
                loss = loss_func(outputs, labels)
                loss.backward()
                formal_optimizer.step()
                if mode == 1:
                    formal_lr_scheduler.step()
                count += 1
                #break
                if count % vis_iter == 0:
                    self.func("iter " + str(count) + "| loss: " + str(loss.item()))
            formal_optimizer.zero_grad()
            if test:
                self.model.eval()
                accuracy = self.test_model(valid_loader)
                self.func("Acc: " + str(accuracy))
                self.model.train()
        if test:        
            self.func("final acc: " + str(accuracy))

    def formal_once(self, train_data, train_label,
                    loss_func = torch.nn.CrossEntropyLoss()):
        self.start_up()
        formal_optimizer = torch.optim.Adam(self.classifier.parameters(),lr=1e-3)
        if self.use_gpu:
            train_data, train_label = train_data.cuda(), train_label.cuda()
        loss = backward(train_data, train_label, self, formal_optimizer, loss_func)
        output = "Data amount:" + len(train_data) + "| Loss: " + loss.item()
        self.func(output)

    def load_prototype(self, filename, mode = "whole", keywords=["fc","classifier"]):
        # Load prototype to get suitable backbone
        # mode: loading mode
        # "whole" save structure and params
        # "params": save parameters
        if self.use_gpu:
            if mode == "whole":
                self.model = nn.DataParallel(torch.load(filename), device_ids=self.use_gpu).cuda()
            else:
                self.model.load_state_dict(self.update_param(torch.load(filename),0))
        else:
            if mode == "whole":
                self.model = torch.load(filename,map_location='cpu')
            else:
                self.model.load_state_dict(self.update_param(torch.load(filename),0))
        keywords = self.keywords
        output = 'Load prototype from ' + filename
        self.func(output)

        self.optional_node, self.avaliable_layers = get_optional_layers(getLayers(self.model), keywords)

    def save_model(self,filename):
        output = 'Save the model as ' + filename
        self.func(output)
        features = []
        for layer in getLayers(self.classifier).values():
            if isinstance(layer,nn.Linear):
                features.append(layer.out_features)
        state = {"net": self.state_dict(), "layers" : self.layers,
                 "mode" : self.mode, "features" : features}
        torch.save(state, filename)

    def load_model(self, filename):
        output = 'Load the model from ' + filename
        self.func(output)
        checkpoint = torch.load(filename)
        if len(self.classifier):
            #delete existing classifier
            self.del_memes()
        self.add_layers(checkpoint['layers'])
        features = checkpoint['features']
        self.making_classifier(tuple(features[:-1]),features[-1])
        self.load_state_dict(self.update_param(checkpoint['net']))
        self.mode = checkpoint['mode']

    def update_param(self, params, index=1):
        names = list(params.keys())
        for name in names:
            parts = name.split(".")
            if self.use_gpu:
                if "module" not in parts:
                    if parts[0] != "classifier":
                        parts.insert(index,"module")
                        params[".".join(parts)] = params.pop(name)
            else:
                if "module" in parts:
                    parts.remove("module")
                    params[".".join(parts)] = params.pop(name)
        return params

    def get_representation(self, images, grad =False):
        if self.use_gpu: images = images.cuda()
        if grad: self.model(images)
        else:
            with torch.no_grad():
                self.model(images)

    def fetch_record(self, info = False):
        self.func("Fetching records...")
        dict = self.get_record()
        if info:
            self.func("Records fetched:")
            for key in dict:
                output = "{} : {}".format(self.optional_node[key],
                    "*".join([str(i) for i in dict[key].shape]))
                self.func(output)
        self.func("current recording status: " + str(self.rec))
        return dict

    def get_record(self,refresh=True):
        record={}
        for name in self.monitor:
            result = self.monitor[name]
            if len(result)!=0:
                if len(self.use_gpu)>1:
                    device_order = []
                    for i in range(len(result)):
                        device_order.append(result[i].device.index + i // len(self.use_gpu) * len(self.use_gpu))
                    used_result = [result[i[0]] for i in sorted(enumerate(device_order), key=lambda x: x[1])]
                else:
                    used_result = result
            record[name] = torch.cat([item for item in used_result])
            if refresh: self.monitor[name] = []
        return record

    def get_name(self,name):
        if self.use_gpu:
            if name[:7] == "module.":
                return name
            else:
                return "module." + name
        else:
            if name[:7] == "module.":
                return name[7:]
            else:
                return name

    def update_name(self, targets, target_channels,
                    target_stride, meme_sizes):
        #check names are insistent
        new_targets = []
        new_channels = {}
        new_stride = {}
        new_sizes = {}
        for target in targets:
            assert target in target_channels
            assert target in target_stride
            assert target in meme_sizes
            real_target = self.get_idx(target)
            if real_target != target:
                output = "Changing layer name:{}——>{}".format(target, real_target)
                self.func(output)
                new_targets.append(real_target)
                new_channels[real_target] = target_channels[target]
                new_stride[real_target] = target_stride[target]
                new_sizes[real_target] = meme_sizes[target]
            else: new_targets.append(target)
        return new_targets, new_channels, new_stride, new_sizes

    def save_prototype(self,filename):
        output = 'Save the backbone as ' + filename
        self.func(output)
        torch.save(self.model.state_dict(), filename)

    def get_params(self):
        layer=self.classifier[0]
        weight = layer.weight.detach().numpy()
        bias = layer.bias.detach().numpy()
        return weight,bias

    def get_diff(self, imgs, layer, channel, start,
                 shape):
        img_tensor = torch.from_numpy(imgs)
        erfs = []
        erfs.append(self.get_diff_batch(img_tensor,
                    layer, channel, start, shape))
        return torch.cat(erfs, dim = 0)

    def get_diff_batch(self, img_tensor, layer,
                       channel, start, shape):
        length = len(img_tensor)
        img_tensor.requires_grad = True

        erf = torch.zeros_like(img_tensor)
        assert self.rec
        self.get_representation(img_tensor, True)
        output = self.get_record()[layer]
        mask = torch.zeros_like(output)
        for i in range(length):
            mask[i, int(channel[i]), int(start[i][0]):int(start[i][0] + shape[i][0]), \
                int(start[i][1]):int(start[i][1] + shape[i][1])] = 1

        output = torch.mean(output * mask, dim=(1, 2, 3))
        kernel = torch.eye(len(img_tensor))
        for i in range(length):
            output.backward(kernel[i], retain_graph=True)
            erf[i] = img_tensor.grad[i]
            img_tensor.grad.zero_()
        del img_tensor
        del kernel
        return erf

    def show_rf(self, name, data, key_dict, save_path="memes",
                  depository=True, history = []):
        #depository: save origional data for further reload
        #mode: different working mode(stackable)
        #heat: Use heat map to show meme classification ability
        #erf: Show effective receptive fields(erf)
        #trf: Use the bounding box to mark the theoretical receptive field(trf)
        self.func('Start visualizing!')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            vis_list = []
        else:
            vis_list = os.listdir(save_path)
        if depository:
            store_path = "depository"
            if not os.path.exists(store_path):
                os.mkdir(store_path)
                store_list = []
            else:
                store_list = os.listdir(store_path)
        for filename in history:
            if filename in vis_list:
                vis_list.remove(filename)
            if filename in store_list:
                store_list.remove(filename)
            elif "L_" + filename in store_list:
                store_list.remove("L_" + filename)
        for layer in key_dict:
            infos = key_dict[layer]
            target = np.array(infos["idx"])
            if len(target) > 0:
                imgs = data[target]
                starts = np.array(infos["start"])
                shapes = np.array(infos["shape"])
                channels = np.array(infos["channel"])
                intensitys = infos["intensity"]
                names = infos["name"]
                del infos, target
                for series, start, shape, channel, intensity, file_name in zip(imgs, starts, shapes, channels, intensitys, names):
                    x = np.arange(len(series))
                    y = series[:,0]
                    plt.plot(x, y)
                    plt.xlabel("time")
                    plt.ylabel(name)
                    plt.axvline(start[0], c = 'r')
                    plt.axvline(start[0]+shape[0], c = 'r')
                    plt.savefig(file_name)
                    plt.close()

                    x = np.arange(shape[0])
                    y = series[start[0]:start[0] + shape[0], 0]
                    plt.plot(x, y)
                    plt.xlabel("time")
                    plt.ylabel(name)
                    memename = os.path.join(store_path, os.path.split(file_name)[1].split(".")[0] + "_meme.png")
                    plt.savefig(memename)
                    plt.close()

                    if depository:
                        fullname = os.path.split(file_name)[1].split(".")[0] + ".txt"
                        storename = os.path.join(store_path, fullname)
                        np.savetxt(storename, series)
                    self.func("Memes saved as {}".format(file_name))
            output = "Delete redundant images."
            self.func(output)
            for filename in vis_list:
                to_del = os.path.join(save_path, filename)
                os.remove(to_del)
            if depository:
                for filename in store_list:
                    to_del = os.path.join(store_path, filename)
                    os.remove(to_del)

    def extend_summary(self, model, in_submodule=0, keywords=["fc", "classifier"], prefix="",
                       intermediate=6, intermediate_text=15, branch_length_limit=21,
                       branch=False, backbone_length_limit=20):
        sublayers = get_sublayers(model)
        layers = list(model.named_children())
        for count in range(len(layers)):
            name, structure = layers[count]
            backbone_name = prefix + name
            idx = self.get_idx(backbone_name)
            if idx in self.hooks:
                left_output = (backbone_name + " (monitored)").center(backbone_length_limit)
            else:
                left_output = backbone_name.center(backbone_length_limit)
            kernel_size = []
            for key in self.memes:
                where, l, w = key.split("_")
                if idx == int(where):
                    kernel_size.append("{}×{}".format(l, w))
            if len(kernel_size):
                used_size = ','.join(kernel_size)
                middle_output = intermediate * '-' + used_size.center(intermediate_text) + intermediate * '-'
                right_output = '+'.center(branch_length_limit)
                branch = True
            elif branch:
                middle_output = (2 * intermediate + intermediate_text) * ' '
                right_output = '|'.center(branch_length_limit)
            else:
                middle_output = ''
                right_output = ''
            self.func(left_output + middle_output + right_output)
            if name in sublayers:
                in_submodule += 1
                branch = self.extend_summary(structure, in_submodule, keywords,
                            prefix=prefix + name + ".", branch=branch)
                in_submodule -= 1
            if in_submodule == 0 and backbone_name != layers[-1][0]:
                left_output = '|'.center(backbone_length_limit)
                if branch:
                    middle_output = (2 * intermediate + intermediate_text) * ' '
                    right_output = '|'.center(branch_length_limit)
                    self.func(left_output + middle_output + right_output)
                else:
                    self.func(left_output)

    def visualizing(self, input_data, kernels={}, func=Standardization, \
                    save_path = "visualization", transparency = 0.2):
        # input: Tensor
        # channel * length * width


        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            self.func("Delete all pics in "+save_path+".")
            files = os.listdir(save_path)
            for file in files:
                filename = os.path.join(save_path,file)
                os.remove(filename)
        input_img = torch.unsqueeze(input_data, 0)
        if not self.rec:
            self.switch_monitoring()
        for name in self.monitor:
            self.monitor[name] = []
        self.forward(input_img)
        self.switch_monitoring()
        img_length = input_img.shape[-2]
        img_width = input_img.shape[-1]
        img = np.transpose(input_data.detach().numpy(),
                           (1, 2, 0))
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 255
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * 255

        cv2.imwrite(os.path.join(save_path,
            "original.jpg"), cv2.resize(img, (1200, 1200)))

        for target in self.dist_maps:
            where, length, width = target.split("_")
            shape = (int(length), int(width))
            where = self.optional_node[int(where)]
            if target not in self.rf.receptive_fields:
                self.rf.add_rf(where, shape)
            used_map = self.gather_map(self.dist_maps[target]).cpu()
            dist_map = used_map[0].detach().numpy()
            quantile = np.percentile(np.max(dist_map,axis=(1,2)), 90)
            print(target)
            print(len(kernels[target]['index']))
            print(len(dist_map))
            for count, target_map in enumerate(dist_map):
                if len(kernels) != 0:
                    label, channel, _, idx = kernels[target]['index'][count]
                    filename = '_'.join(
                            [str(label), where, str(channel), str(idx + 1)]) + '.jpg'
                else:
                    filename = where + '_' + str(count) + '.jpg'

                if self.mode == 'sim':
                    thres = 0.2
                else:
                    if len(kernels) != 0:
                        threshold = kernels[target]['thresholds'][count]
                    else:
                        threshold = 0.5
                    std = func(threshold)
                    target_map = std(target_map)
                    thres = std.res
                    quantile = std(quantile)

                x_coord, y_coord = self.rf.get_coord(where, target_map.shape)
                pad_dist_map = np.pad(target_map, ((1, 1), (1, 1)), 'constant', constant_values=0)
                pad_x_coord = np.pad(x_coord, (1, 1), 'constant', constant_values=(0, img_length))
                pad_y_coord = np.pad(y_coord, (1, 1), 'constant', constant_values=(0, img_width))

                if len(x_coord) == 1 or len(y_coord) == 1:
                    interpfunc = interpolate.interp2d(pad_y_coord, pad_x_coord, pad_dist_map, kind='linear')
                else:
                    interpfunc = interpolate.interp2d(pad_y_coord, pad_x_coord, pad_dist_map, kind='cubic')

                zn = interpfunc(np.arange(img_length), np.arange(img_width))

                regions = crop_bbox(zn, thres, times = 2)

                if len(regions) != 0:
                    bound_img = img.copy()
                    crop = False
                    for region in regions:
                        left_upper, right_lower, amount = region
                        if np.max(target_map) > quantile and self.rf.limiter(where, left_upper, right_lower, amount):
                            cv2.rectangle(bound_img, (left_upper[1], left_upper[0]), (right_lower[1], right_lower[0]),
                                              (0, 255, 0))
                            bound_pic = np.uint8(np.clip(bound_img, 0, 255))
                            crop = True
                    if crop:
                        boundname = os.path.join(save_path, filename)
                        cv2.imwrite(boundname, cv2.resize(bound_pic, (1200, 1200)))
                        self.func(boundname, "saved")
                heatmap = cv2.applyColorMap(np.uint8(np.clip(255 * zn, 0, 255)), cv2.COLORMAP_JET)
                overlayed_img = transparency * img + (1 - transparency) * heatmap
                heatname = os.path.join(save_path, "heat_" + filename)
                cv2.imwrite(heatname, cv2.resize(overlayed_img, (1200, 1200)))
                self.func(heatname, "saved")

    def visualizing_v2(self, input_data, input_labels, kernels={}, func=Standardization, \
                    save_path = "visualization", transparency = 0.2,  labels = [], k = 5,
                    ref_path = "memes"):
        # input: Tensor
        # channel * length * width
        assert len(self.classifier) == 1
        weight = self.classifier.classifier.weight
        classes, _ = weight.shape
        if len(labels) != classes:
            labels = [str(i) for i in range(classes)]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            self.func("Delete all pics in " + save_path + ".")
            files = os.listdir(save_path)
            for file in files:
                filename = os.path.join(save_path,file)
                os.remove(filename)

        input_img = torch.unsqueeze(input_data, 0)
        if not self.rec:
            self.switch_monitoring()
        for name in self.monitor:
            self.monitor[name] = []
        features = self.get_features(input_img)
        output = self.classifier(features).cpu()
        _, preds = torch.max(output, 1)
        self.switch_monitoring()
        if input_labels != preds[0]:
            fres = torch.mul(features, weight[preds])
            _, fp_idx = fres.topk(k=k, dim=1)
            fp_idx = fp_idx.cpu().numpy()[0]
        else:
            fp_idx = []
        tres = torch.mul(features, weight[input_labels])
        _, tp_idx = tres.topk(k=k, dim=1)
        _, tn_idx = tres.topk(k=k, dim=1, largest=False)
        tp_idx = tp_idx.cpu().numpy()[0]
        tn_idx = tn_idx.cpu().numpy()[0]

        img_length = input_img.shape[-2]
        img_width = input_img.shape[-1]
        img = np.transpose(input_data.detach().numpy(),
                           (1, 2, 0))
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 255
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * 255

        cv2.imwrite(os.path.join(save_path,
            "original_"+labels[input_labels]+".jpg"), cv2.resize(img, (1200, 1200)))
        cnt = 0
        for target in self.dist_maps:
            where, length, width = target.split("_")
            shape = (int(length), int(width))
            where = self.optional_node[int(where)]
            if target not in self.rf.receptive_fields:
                self.rf.add_rf(where, shape)
            used_map = self.gather_map(self.dist_maps[target]).cpu()
            dist_map = used_map[0].detach().numpy()
            quantile = np.percentile(np.max(dist_map,axis=(1,2)), 95)
            for count, target_map in enumerate(dist_map):
                state = False
                if cnt in tp_idx:
                    to_save = os.path.join(save_path, "tp")
                    state = True
                if cnt in tn_idx:
                    to_save = os.path.join(save_path, "tn")
                    state = True
                if cnt in fp_idx:
                    to_save = os.path.join(save_path, "fp")
                    state = True
                if state:
                    if not os.path.exists(to_save):
                        os.makedirs(to_save)
                    if len(kernels) != 0:
                        label, channel, _, idx = kernels[target]['index'][count]
                        filename = '_'.join(
                                [labels[label], where, str(channel), str(idx)]) + '.jpg'
                    else:
                        filename = where + '_' + str(count) + '.jpg'

                    if self.mode == 'sim':
                        thres = quantile
                    else:
                        if len(kernels) != 0:
                            threshold = kernels[target]['thresholds'][count]
                        else:
                            threshold = 0.5
                        std = func(threshold)
                        target_map = std(target_map)
                        thres = std.res
                        quantile = std(quantile)
                    if np.std(target_map) < np.mean(target_map) ** 2: continue
                    x_coord, y_coord = self.rf.get_coord(where, target_map.shape)

                    pad_dist_map = np.pad((target_map/np.max(target_map)) ** 2, ((1, 1), (1, 1)), 'constant', constant_values=0)
                    pad_x_coord = np.pad(x_coord, (1, 1), 'constant', constant_values=(0, img_length))
                    pad_y_coord = np.pad(y_coord, (1, 1), 'constant', constant_values=(0, img_width))

                    if len(x_coord) == 1 or len(y_coord) == 1:
                        interpfunc = interpolate.interp2d(pad_y_coord, pad_x_coord, pad_dist_map, kind='linear')
                    else:
                        interpfunc = interpolate.interp2d(pad_y_coord, pad_x_coord, pad_dist_map, kind='linear')

                    zn = interpfunc(np.arange(img_length), np.arange(img_width))

                    regions = crop_bbox(zn, thres, 1)

                    if len(regions) != 0:
                        bound_img = img.copy()
                        crop = False
                        for region in regions:
                            left_upper, right_lower, amount = region
                            cv2.rectangle(bound_img, (left_upper[1], left_upper[0]), (right_lower[1], right_lower[0]),
                                                  (0, 255, 0))
                            bound_pic = np.uint8(np.clip(bound_img, 0, 255))
                            crop = True
                        if crop:
                            boundname = os.path.join(to_save, filename)
                            cv2.imwrite(boundname, cv2.resize(bound_pic, (1200, 1200)))
                            self.func(boundname, "saved")
                    heatmap = cv2.applyColorMap(np.uint8(np.clip(255 * zn, 0, 255)), cv2.COLORMAP_JET)
                    overlayed_img = transparency * img + (1 - transparency) * heatmap
                    heatname = os.path.join(to_save, "heat_" + filename)

                    to_read = kernels[target]['name'][count]
                    if os.path.exists(to_read):
                        img1 = cv2.imread(to_read)
                        img2 = cv2.resize(overlayed_img, (1200, 1200))
                        cv2.imwrite(heatname, np.hstack((img1, img2)))
                    else:
                        print("Visualizations of memes will provide vivid figures.")
                        cv2.imwrite(heatname, cv2.resize(overlayed_img, (1200, 1200)))
                    self.func(heatname, "saved")
                cnt += 1

    def fetch_memes(self, used_channels, used_stride, used_size, num_channels,
                    train_data, train_label, zscore, prune_iter, memes, keep_channel, interval):


        self.switch_monitoring()
        self.get_representation(train_data)

        map_dict = self.fetch_record(info=True)
        # zscore.cpu()
        zscore.update(map_dict, True)
        used_label = train_label.detach().numpy()

        memes.get_memes_from_field(search_dict=map_dict, search_label=used_label, compare_dict=map_dict,
                                   compare_label=used_label, target_channels=used_channels,
                                   target_stride=used_stride, num_channels=num_channels,
                                   sizes=used_size, times=interval,
                                   prune_iter=prune_iter, 
                                   optional_nodes=self.optional_node)

        memes.meme_prune(keep_channel=keep_channel)

        self.switch_monitoring()