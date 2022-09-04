# EMNLP 2022 Submission - ECTSum: A New Benchmark Dataset For Bullet Point Summarization of Long Earnings Call Transcripts

## Note To The Reviewers:
A few links given in the paper are not working unfortunately. Please find the proper links below:
- Page 3 Link 9
  - Existing Link: shorturl.at/ajkvQ (Issue with short url app)
  - Correct Link: https://www.forbes.com/sites/forbesfinancecouncil/2018/07/13/the-six-most-important-takeaways-from-any-quarterly-earnings-report/
- Page 3 Link 10
  - Existing Link: https://www.refinitiv.com/en/financial-data/company (Half of the link got broken into Page 4)
  - Correct Link: https://www.refinitiv.com/en/financial-data/company-data/ibes-estimates
- Page 8 Link 16
  - Existing Link: shorturl.at/esvxX (Issue with short url app)
  - Correct Link: https://www.fool.com/earnings/call-transcripts/2021/08/04/fleetcor-technologies-inc-flt-q2-2021-earnings-cal/
  
## Dataset
A sample of our dataset can be found in the folder `dataset`. <br />
The folders `dataset > train`, `dataset > val`, and `dataset > test` respectively contain 35, 5, and 10 document-summary pairs reflecting the split ratio of 70:10:20 used to prepare the whole dataset.

## Codes
Codes for our baselines and proposed model **ECT-BPS** can be found under `codes`

## Survey
We give below the links to some of the survey forms which were provided to the financial experts for the human evaluation experiment:
- Form 1: https://forms.gle/y5xg7VuqqtPiLLY39
- Form 2: https://forms.gle/rKVDKz8TiWSNn8kK9
- Form 3: https://forms.gle/pWtexZqM9TXGGoCAA

## Updated Metrics
<p align="center"><img src="https://github.com/rajdeep345/ECTSum/blob/main/figures/updated_figure_3.png" width="1380px" height="180px"></p>
Updated Figure 3: Histogram distribution for human evaluation scores assigned to model-generated summaries
