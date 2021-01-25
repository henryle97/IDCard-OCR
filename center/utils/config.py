import os
import yaml


class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fpath):
        if not os.path.exists(fpath):
            print("Not exists config path: ", fpath)
            return None

        with open(fpath, encoding='utf-8') as f:
            config =yaml.safe_load(f)

        config = Cfg.update_config(config)

        return config

    def save(self, f_out_path):
        with open(f_out_path, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def update_config(config):
        """
        return new config
        :param config:
        :return: new config
        """
        config['model']['output_h'] = config['model']['input_h'] // config['model']['down_ratio']
        config['model']['output_w'] = config['model']['input_w'] // config['model']['down_ratio']
        config['gpu_str'] = config['gpus']

        config['gpus'] = [int(gpu) for gpu in config['gpus'].split(',')]
        config['gpus'] = [i for i in config['gpus']] if config['gpus'][0] >=0 else [-1]
        if config['gpus'] != '-1':
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in config['gpus'])

        if config['train']['master_batch_size'] == -1:
            config['train']['master_batch_size'] = config['train']['batch_size'] // len(config['gpus'])

        rest_batch_size = (config['train']['batch_size'] - config['train']['master_batch_size'])
        chunk_sizes = [config['train']['master_batch_size']]
        for i in range(len(config['gpus']) - 1):
            slave_chunk_size = rest_batch_size // (len(config['gpus']) - 1)
            if i < rest_batch_size % (len(config['gpus']) - 1):
                slave_chunk_size += 1
            chunk_sizes.append(slave_chunk_size)
        config['train']['chunk_sizes'] = chunk_sizes

        config['root_dir'] = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

        config['exp_dir'] = os.path.join(config['root_dir'], 'exp')
        config['save_dir'] = os.path.join(config['exp_dir'], config['exp_id'])
        config['debug_dir'] = os.path.join(config['save_dir'], 'debug')
        print('The output model will be saved to ', config['save_dir'])

        return config
if __name__ == "__main__":
    config = Cfg.load_config_from_file("/home/hisiter/working/CMND/Centernet_custom_v3/center/config/base.yml")
    a = config['model']['heads']

    print(config)
    print(config['data_dir'])




