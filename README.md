# ECTSum: A New Benchmark Dataset For Bullet Point Summarization of Long Earnings Call Transcripts

Long Paper Accepted at the <b> EMNLP 2022 Main Conference! </b> <br /> 
<li> ArXiv Preprint: https://arxiv.org/pdf/2210.12467.pdf </li>
<li> Poster: https://rajdeep345.github.io/files/pdf/research/ECTSum_EMNLP2022_Poster.pdf </li>
<li> Pre-recorded Video: https://drive.google.com/file/d/1DW2i2ApgiE6V7ViiayX5zdJSRXdAEbsy/view </li>

## Dataset
The <b> <i> ECTSum </b> </i> dataset can be found under the `data` folder.

## Codes
Codes and instructions for our proposed model <b> <i> ECT-BPS </b> </i> can be found under `codes/ECT-BPS` <br />
Codes and instructions for our baseline models can be found under `codes/baselines`

## Data Preparation for ECT-BPS
### Preparing the data for training the <i> Extractive Module </i>

#### Imports
`pip install sentence-transformers` </br>
`pip install num2words` </br>
`pip install word2number` </br>

#### Prepare the data
`python prepare_data_ectbps_ext.py`

#### Data Location
The data is saved at `codes/ECT-BPS/ectbps_ext/data/`. </br>
Processed data is already uploaded at this location.

## Updates
<li> 1st November 2022 - ECTSum Dataset released </li>
<li> 30th November 2022 - Codes and Instructions released for training the Extractive Module of ECT-BPS
