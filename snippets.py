# class TensorboardCallback(Callback):
#     def __init__(self,tensorboard_writer,creation_time,lr_sequence,with_input=False,
#                  with_loss=True, save_preds=False, save_targs=False, concat_dim=0):
#         store_attr(self, "with_input,with_loss,save_preds,save_targs,concat_dim")
#         self.tensorboard_writer=tensorboard_writer
#         self.count=0
#         self.creation_time = creation_time
#         self.lr_sequence=lr_sequence
#
#     def begin_batch(self):
#         if self.with_input: self.inputs.append((to_detach(self.xb)))
#
#     def begin_validate(self):
#         "Initialize containers"
#         # self.preds,self.targets = [],[]
#         # if self.with_input: self.inputs = []
#         if self.with_loss:
#             self.losses = []
#             self.accuracy=[]
#
#     def after_batch(self):
#         if not self.training:
#             "Save predictions, targets and potentially losses"
#
#             # preds,targs = to_detach(self.pred),to_detach(self.yb)
#             # if self.save_preds is None: self.preds.append(preds)
#             # else: (self.save_preds/str(self.iter)).save_array(preds)
#             # if self.save_targs is None: self.targets.append(targs)
#             # else: (self.save_targs/str(self.iter)).save_array(targs[0])
#             if self.with_loss:
#                 self.accuracy.append(self.metrics[0].value)
#                 self.losses.append(to_detach(self.loss))
#     def after_validate(self):
#         "Concatenate all recorded tensors"
#         # if self.with_input:     self.inputs  = detuplify(to_concat(self.inputs, dim=self.concat_dim))
#         # if not self.save_preds: self.preds   = detuplify(to_concat(self.preds, dim=self.concat_dim))
#         # if not self.save_targs: self.targets = detuplify(to_concat(self.targets, dim=self.concat_dim))
#         if self.recorder.metric_names.index("accuracy")<len(self.recorder.log):
#             self.tensorboard_writer.add_scalar(str(self.lr_sequence)+" loss: ",self.recorder.log[self.recorder.metric_names.index("valid_loss")],self.count)
#             self.tensorboard_writer.add_scalar(str(self.lr_sequence)+" accuracy: ",self.recorder.log[self.recorder.metric_names.index("accuracy")],self.count)
#             self.count+=1
#
#     def all_tensors(self):
#         res = [None if self.save_preds else self.preds, None if self.save_targs else self.targets]
#         if self.with_input: res = [self.inputs] + res
#         if self.with_loss:  res.append(self.losses)
#         return res
# tensorboardcb = TensorboardCallback(writer,creation_time,lr_sequence)

# def _apply_cats (voc, add, c):
#     if not is_categorical_dtype(c):
#         return pd.Categorical(c, categories=voc[c.name][add:]).codes+add
#     return c.cat.codes+add #if is_categorical_dtype(c) else c.map(voc[c.name].o2i)
# def _decode_cats(voc, c): return c.map(dict(enumerate(voc[c.name].items)))
#
# class Categorify():
#     "Transform the categorical variables to something similar to `pd.Categorical`"
#     order = 1
#
#     def setups(self, to,class_groups=class_groups):
#
#         self.classes = {n:class_groups[n[n.rfind("_")+1:]] for n in to.cat_names}
#         self.class_groups = class_groups
#
#
#     def encodes(self, to): to.transform(to.cat_names, partial(_apply_cats, self.classes, 0))
#     def decodes(self, to): to.transform(to.cat_names, partial(_decode_cats, self.classes))
#     def __getitem__(self,k): return self.classes[k]