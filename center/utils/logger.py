from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
from .config import Cfg

USE_TENSORBOARD = True
try:
  import tensorboardX
  print('Using tensorboardX')
except:
  USE_TENSORBOARD = False

class Logger(object):
  def __init__(self,  config):
    """Create a summary writer logging to log_dir."""
    save_dir = config['save_dir']
    debug_dir = config['debug_dir']
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    if not os.path.exists(debug_dir):
      os.makedirs(debug_dir)
   
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    args = dict((name, getattr(config, name)) for name in dir(config)
                if not name.startswith('_'))
    file_name = os.path.join(save_dir, 'opt.yaml')

    Cfg(config).save(file_name)
          
    log_dir = save_dir + '/logs_{}'.format(time_str)
    if USE_TENSORBOARD:
      self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
    else:
      if not os.path.exists(os.path.dirname(log_dir)):
        os.mkdir(os.path.dirname(log_dir))
      if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    self.log = open(log_dir + '/log.txt', 'w')
    try:
      os.system('cp {}/opt.yaml {}/'.format(save_dir, log_dir))
    except:
      pass
    self.start_line = True

  def write(self, txt):
    if self.start_line:
      time_str = time.strftime('%Y-%m-%d-%H-%M')
      self.log.write('{}: {}'.format(time_str, txt))
    else:
      self.log.write(txt)  
    self.start_line = False
    if '\n' in txt:
      self.start_line = True
      self.log.flush()
  
  def close(self):
    self.log.close()
  
  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    if USE_TENSORBOARD:
      self.writer.add_scalar(tag, value, step)


if __name__ == "__main__":
  pass