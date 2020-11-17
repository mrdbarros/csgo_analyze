import data_loading
import torch





def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat ** 0.56))


class LinBnDrop(torch.nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"

    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [torch.nn.BatchNorm1d(n_in)] if bn else []
        if p != 0: layers.append(torch.nn.Dropout(p))
        lin = [torch.nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)



class TabularModelCustom(torch.nn.Module):
    "Basic model for tabular data."

    def __init__(self, category_list, class_groups_sizes, n_cont, layers, ps=None, embed_p=0.,
                 use_bn=True, bn_final=True, bn_cont=True):

        super().__init__()
        ps = ps

        class_group_map = {}
        for i, cat in enumerate(category_list):
            class_group = cat[cat.rfind("_") + 1:]
            class_group_index, _ = class_groups_sizes[class_group]
            if class_group_index in class_group_map:
                class_group_map[class_group_index].append(i)
            else:
                class_group_map[class_group_index] = [i]
        self.class_group_map = class_group_map
        self.embeds = torch.nn.ModuleList(
            [torch.nn.Embedding(index_ni[1], emb_sz_rule(index_ni[1])) for _, index_ni in class_groups_sizes.items() if
             index_ni[1] > 2])
        self.emb_drop = torch.nn.Dropout(embed_p)
        self.bn_cont = torch.nn.BatchNorm1d(n_cont) if bn_cont else None

        binary_size = sum(len(class_group_map[i]) for i in range(len(self.embeds), len(class_group_map)))
        n_emb = sum(e.embedding_dim * len(class_group_map[i]) for i, e in enumerate(self.embeds)) + binary_size
        self.n_emb, self.n_cont = n_emb, n_cont
        sizes = [n_emb + n_cont] + layers
        actns = [torch.nn.ReLU(inplace=True) for _ in range(len(sizes) - 2)]
        _layers = [LinBnDrop(sizes[i], sizes[i + 1], bn=use_bn and (i != len(actns) - 1 or bn_final), p=p, act=a)
                   for i, (p, a) in enumerate(zip(ps, actns))]

        self.layers = torch.nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):

        if self.n_emb != 0:
            x_cat_binary = []
            for i in range(len(self.embeds), len(self.class_group_map)):
                x_cat_binary += self.class_group_map[i]
            with torch.no_grad():
                x_cat_binary = x_cat[:, x_cat_binary].float()
            x_cat_nonbinary = [torch.flatten(e(x_cat[:, self.class_group_map[i]]), start_dim=1) for i, e in
                               enumerate(self.embeds)]
            x = torch.cat(x_cat_nonbinary + [x_cat_binary], 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)



class CustomMixedModel(torch.nn.Module):
    def __init__(self, image_model, tab_model, seq_model, image_output_size, embeds_size,prepare_and_pad,max_image_batch):
        super(CustomMixedModel, self).__init__()
        self.image_model = image_model
        # embedding types are primaries, secondaries, flashbangs and binaries
        # self.classifier = TabularModel_NoCat(emb_sizes,1536, 30,[400],ps=[0.1],use_bn=False)
        self.tab_model = tab_model
        # n_emb = sum(e.embedding_dim for e in self.embeds)
        self.seq_model = seq_model
        self.classifier = torch.nn.Sequential(LinBnDrop(200 + image_output_size
                                                        + embeds_size
                                                        , 1, act=None, p=0.))
        self.prepare_and_pad = prepare_and_pad
        self.max_image_batch = max_image_batch
    def forward(self, input_cat, input_cont, input_image, attention_mask, train_embeds=True, train_seq_model=True):
        valid_sizes = torch.sum((attention_mask == 1), dim=1)
        if train_embeds:
            input_embed = self.forward_embeds(input_cat, input_cont, input_image,valid_sizes)
        else:
            with torch.no_grad():

                input_embed = self.forward_embeds(input_cat, input_cont, input_image,valid_sizes)

        input_embed=self.prepare_and_pad(input_embed,valid_sizes)

        if train_seq_model:

            bert_out = self.forward_seq_model(input_embed, attention_mask)
        else:
            with torch.no_grad():
                bert_out = self.forward_seq_model(input_embed, attention_mask)



        output = self.classifier(
            torch.cat((input_embed[range(input_embed.shape[0]), valid_sizes-1], bert_out),
                      dim=1))
        # output = self.classifier(input_embed[range(input_embed.shape[0]), (input_embed.shape[1] - mask_size - 1)])
        # output = self.classifier(bert_out)
        return output

    def forward_embeds(self, input_cat, input_cont, input_image,valid_size):
        n_batches = (input_image.shape[0]//self.max_image_batch)+1*(input_image.shape[0]%self.max_image_batch>0)

        tab_out = self.tab_model(input_cat,input_cont)
        image_out = torch.cat([self.image_model(input_image[i*self.max_image_batch:min((i+1)*self.max_image_batch,input_image.shape[0])])
                                 for i in range(n_batches)],dim=0)
        # comprehension to break inputs in 'n_batches' of 'self.max_image_batch' size
        input_embed = torch.cat((tab_out,image_out),dim=1)
        input_embed = torch.nn.ReLU()(input_embed)
        return input_embed

    def forward_seq_model(self, input_embed, attention_mask):
        # bert_out = self.seq_model(input_embed.permute((1, 0, 2)),
        #                           src_key_padding_mask=attention_mask).permute((1, 0, 2))[:, 0]
        bert_out = torch.mean(self.seq_model(inputs_embeds=input_embed,
                                  attention_mask=attention_mask)[0],dim=1)
        bert_out = torch.nn.ReLU()(bert_out)
        return bert_out


class CustomMixedModelSingleImage(torch.nn.Module):
    def __init__(self, image_model, tab_model, image_output_size,class_p):
        super(CustomMixedModelSingleImage, self).__init__()
        self.image_model = image_model
        # embedding types are primaries, secondaries, flashbangs and binaries
        # self.classifier = TabularModel_NoCat(emb_sizes,1536, 30,[400],ps=[0.1],use_bn=False)
        self.tab_model = tab_model
        # n_emb = sum(e.embedding_dim for e in self.embeds)
        self.classifier = torch.nn.Sequential(LinBnDrop(200 + image_output_size, 1, act=None, p=class_p))

    def forward(self, input_cat, input_cont, input_image):
        output_tabular = self.tab_model(input_cat, input_cont)
        output_image = self.image_model(input_image)
        logits = self.classifier(torch.nn.ReLU()(torch.cat((output_tabular, output_image), dim=1)))

        return logits

