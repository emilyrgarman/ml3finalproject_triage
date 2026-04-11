# Real World Weight Crosentropy Loss Function
# Based on the paper by Ho, Y., & Wookey, S. (2020). 
# The Real-World-Weight Cross-Entropy Loss Function: Modeling the Costs of Mislabeling. 
# IEEE Access, 8, 4806–4813. https://doi.org/10.1109/ACCESS.2019.2962617
# and the authors' implementation at
# https://github.com/yaoshiang/The-Real-World-Weight-Crossentropy-Loss-Function/
#
# Original Copyright: 
# Copyright (C) 2019 Yaoshiang Ho
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# Contact author for exceptions.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch # Converted to pytorch
# import tensorflow
# from tensorflow.keras import backend as K
from typing import Union

def create_rww_categorical_crossentropy(k, 
                                        loss_type, 
                                        fn_weights:Union[np.ndarray, None]=None, 
                                        fp_weights:Union[np.ndarray, None]=None, 
                                        return_weights=False):
  """Real-World-Weighted crossentropy between an output tensor and a target tensor.
  
  The loss_types other than rww_categorical_crossentropy reimplement existing 
  functions in Keras but are not as well optimized. 
  These loss_types are usable directly, but, are more useful when calling 
  return_weights=True, which then returns fn and fp weights matrixes of size (k,k). 
  Editing those to reflect real world costs, then passing them back into 
  create_rww_crossentropy with loss_type "rww_crossentropy" is the recommended approach. 

  Example Usage: 

  Suppose you have three classes: cat, dog, and other.
  
  Cat is one-hot encoded as [1,0,0], dog as [0,1,0], other as [0,0,1]
  
  The the following code increases the incremental penalty of 
  mislabeling a true target 0 (cat) with a false label 1 (dog) at a cost of 99, 
  versus the default of zero. Note that the existing fn_weights also has a 
  default cost of 1 for missing the true target of 1, for a total cost of 
  100 versus the default cost of 1. 
  
  fn_weights, fp_weights = create_rww_categorical_crossentropy(10, "categorical_crossentropy", return_weights=True)
  fp_weights[0, 1] = 99
  loss = create_rww_categorical_crossentropy(10, "rww_crossentropy", fn_weights, fp_weights)

... 
  
  The fn and fp weights are easy to reason about. 
  
  fn_weights is [x1, __, __]
                [__, x2, __]
                [__, __, x3]
 
  x1 represents the scale of the cost for a fn for cat, x2 for dog, and x3 for other.
  
  This is calculated as fn_weight * log(y_pred). 
  
  In the case of loss_type=categorical_crossentropy, 
  x1, x2, and x3 all equal the value one. 
  All elements not on the main axis must equal zero. 
  
  Note that fn_weights could have been represented as a vector, 
  not a matrix, however, we use a matrix to keep symmetry with 
  fp_weights, and, to prepare for 
  multi-label classification. 
    
  ...

  fp_weights is concerned with the costs of the fps from the other classes. 

  fp_weights of [__, x1, x2]
                [x3, __, x4]
                [x5, x6, __]
 
  x1 represents the cost of predicting 1 for dog, when it should be 0 for cat. 
  x2 represents predicting 2 for other, when the target is 0 for cat. 
  x3 represents predicing 0 for cat, when the target is 1 for dog.
  etc. 
  
  Args:
    * k: 2 or more for number of categories, including "other". 
    * loss_type: "categorical_crossentropy" to initialize to 
      standard softmax_crossentropy behavior, 
      or "weighted_categorical_crossentropy" for standard behavior, or, 
      or "rww_crossentropy" for full weight matrix of all possible fn/fp combinations. 
    * fn_weights: a numpy array of shape (k,k). The main diagonal can
      contain non-zero values; all other values must be zero. 
    * fp_weights: a numpy array of shape (k,k) to define specific combinations 
      of false positive. The main diag should be zeros. 
    * return_weights: If False (default), returns cost function. If True, 
      returns fn and fp weights as np.array. 
Returns:
    * retval: Loss function for use Keras.model.fit, or if return_weights
      arg is True, the fn_weights and fp_weights matrixes. 
  """

  assert return_weights or (fn_weights is not None and fp_weights is not None)

  full_fn_weights = None
  full_fp_weights = None

  anti_eye = np.ones((k,k)) - np.eye(k)
    
  if (return_weights or loss_type=="categorical_crossentropy"):
    full_fn_weights = np.identity((k))
    full_fp_weights = np.zeros((k, k)) # Softmax crossentropy ignores fp.

  elif(loss_type=="weighted_categorical_crossentropy"):
    assert fn_weights is not None
    full_fn_weights = np.eye(k) * fn_weights
    full_fp_weights = np.zeros((k, k)) # softmax crossentropy ignores fp
    
  elif(loss_type=="rww_crossentropy"):
    assert fn_weights is not None
    assert fp_weights is not None
    assert not np.count_nonzero(fn_weights * anti_eye)
    assert not np.count_nonzero(fp_weights * np.eye(k))

    full_fn_weights = fn_weights
    # Novel piece: allow any combination of fp.
    full_fp_weights = fp_weights
    
  else:
    raise Exception("unknown loss_type: " + str(loss_type))
   
  fn_wt = torch.as_tensor(full_fn_weights) # (k,k), always sparse along main diag. 
  fp_wt = torch.as_tensor(full_fp_weights) # (k,k), always dense except main diag. 

  eps = torch.finfo(torch.float64).eps

  def loss_function(output, target):
    print(f'output.shape: {output.shape}, target.shape: {target.shape}')
    output = torch.clamp(output, eps, 1 - eps) 
    
    logs = torch.log(output) # shape (m, k), dense. 1 is good. 
    logs_1_sub = torch.log(1-output) # shape (m, k), dense. 0 is good. 

    m_full_fn_weights = torch.matmul(target, fn_wt) # (m,k) . (k, k)
    m_full_fp_weights = torch.matmul(target, fp_wt) # (m,k) . (k, k)

    return - torch.mean(m_full_fn_weights * logs + 
                    m_full_fp_weights * logs_1_sub)
  
  if (return_weights):
    return full_fn_weights, full_fp_weights
  else:
    return loss_function