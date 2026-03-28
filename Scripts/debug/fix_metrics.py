"""Add wandb and tensorboard to requirements.txt"""
import codecs

f = r'd:\Projects\AI-Projects\EEG\requirements.txt'
c = codecs.open(f, 'r', 'utf-16').read()

if 'wandb' not in c:
    c = c.rstrip() + '\nwandb>=0.16.0\ntensorboard>=2.14.0\n'
    codecs.open(f, 'w', 'utf-16').write(c)
    print('Added wandb + tensorboard')
else:
    print('Already present')
