This is less of a feature request, and more along the lines of: observations of issues I've had when working with the mmdet API and suggestions for how some of these might be refactored to improve the overall package. 

One of the biggest challenges of with working with mmdet so far has been its widespread use of positional arguments. This comes in two flavors: function signatures and return values.


### The current structure

As an example consider `forward_train` function in `base_dense_head.py` and its use of the `get_bboxes` function:


The signature for `get_boxes` looks like:
```python
def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg=None, rescale=False):
    ...
```

And the head forward function looks somewhat like this:

```python
    def forward(self, x):
        # do stuff to translate backbone features into box+scores
        return cls_scores, bbox_preds
```



The `forward_train` function currently looks something like this:

```python
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
```


Imagine if you want to extend `self.forward` to return anything other than a
tuple `Tuple[cls_scores, bbox_preds]`. You have to create a custom `get_boxes`
function that has arguments in an order that agree with the disparate forward
function. Perhaps for some people thinking in terms of ordered items is easy,
but for me, this is incredibly hard to manage. I would like to suggest an
alternative.



### The alternative proposal

Imagine if instead of returning a tuple the `forward` function returned a
dictionary where the keys were standardized instead of the positions of the
values.

```python
    def forward(self, x):
        # do stuff to translate backbone features into box+scores
        outs = {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
        }
        return outs
```

Now, the `get_bboxes` function doesn't need to care about what particular head
was used. It can simply accept the `output` dictionary and assert that it
contains particular keys that that variant of `get_bboxes` needs. (Note this
might allow the head to produce other auxiliary information used in the loss,
but not in the construction of boxes)

```python
def get_bboxes(self, output, img_metas, cfg=None, rescale=False):
    # This get_bboxes function requires cls_scores and bbox_pred, but 
    # its ok if your network produces other things as well
    cls_scores = output['cls_scores']
    bbox_preds = output['bbox_preds']
    # output may have an item 'keypoint_preds' but we aren't forced to care
    # about it if we don't specifically need it.
    ...
```

We can extend this pattern further, so in addition to the `forward` function
producing a dictionary, the `forward_train` function will produce a dictionary
as well.

```python
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        losses = self.loss(outs, gt_bboxes=gt_bboxes, gt_labels=gt_labels,
                           img_metas=img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        train_outputs = {
            'losses': losses,
        }
        if proposal_cfg is not None:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            train_outputs['proposal_list'] = proposal_list

        return train_outputs
```

This has less conditionals and a consistent return type. 

This means that function that use forward train can seamlessly switch between
setting `proposal_cfg` and getting the boxes or just getting the loss because
the return value have consistent types and access patterns in both modes. If you 
do need a conditional it can be based on the return value instead of having to
remember the inputs. 

We could go even further and abstract the labels into a argument called `truth`
that should contain keys: `gt_bboxes`, and optionally `gt_labels` and
`gt_bboxes_ignore`, and perhaps that might look like:


```python
    def forward_train(self, x, img_metas, truth, proposal_cfg=None, **kwargs):
        outs = self.forward(x)
        losses = self.loss(outs, img_metas=img_metas, truth=truth)
        train_outputs = {
            'losses': losses,
        }
        if proposal_cfg is not None:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            train_outputs['proposal_list'] = proposal_list
        return train_outputs
```

### Discussion

IMO this pattern produces much more readable and extensible code. We can:

* Return arbitrary outputs from our forward function

* Add arbitrary target information to the truth dictionary and conditionally
  handle it in our custom loss. 

* Use simpler calling patterns that explicitly extract information from
  returned containers based on standard (easy for humans to remember) string
  keywords rather than standard (hard for humans to remember) integer positions. 

* Use semantically meaningful labels to allow for easier introspection of our
  code at runtime.

I think having a standard set of keywords is much more extensible
than sticking to positional based arguments.

There is a small issue of speed. Unpacking dictionaries is slower than
unpacking tuples, but I don't think it will be noticeable difference given that
every python attribute lookup is a dictionary lookup anyway. 

This is a rather large API change, but I think the reliance of positional based
arguments is stifling further development of new and exciting networks. I think
there might be a way to gradually introduce these changes such that it
maintains a decent level of backwards compatibility as well, but I'm not 100%
sure on this.

I've played around with variants of this and it works relatively well, the main
issue I had was the widespread use of `multi_apply`, which could likely be
changed to assume the containers returned by forward functions are dictionaries
instead of tuples. 


### Conclusion

In summary I want to propose replacing positional based APIs with keyword based
APIs. The main purpose of making this issue is for me to gauge the interest of
the core devs. If there is interest I may work on developing this idea further
and look into implementations that are amenable to a smooth transition such
that backwards compatibility is not broken.
