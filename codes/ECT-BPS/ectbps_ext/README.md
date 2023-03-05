## Training

`python main.py -batch_size 1 -acc_steps 4 -epochs 4 -model RNN_RNN -seed 43`

The model gets saved with the name 'RNN_RNN_seed_{seed}.pt'. 

## Testing

`python main.py -test -batch_size 1 -load_dir checkpoints/RNN_RNN_seed_43.pt`

Load your model according to the seed value used during training.

## Generating the Extractive Summary for a single input ECT file (<i> Prediction Pipeline </i>)

`python main.py -predict -filename 'A_q4_2021.txt' -batch_size 1 -load_dir checkpoints/RNN_RNN_seed_43.pt`

The output <i> extractive summary </i> will be stored at `outputs/hyp/A_q4_2021_ext_summary.txt`
