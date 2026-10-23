# Human rating sheet

100 model reviews, 4 raters (column `rater`), ~25 each.

Fill the five rating columns in `ratings.csv`:

| column | meaning |
|---|---|
| `helpful_1to5` | As the PR author, would this comment help you? 1 = noise, 3 = fair point, 5 = I would act on it |
| `severity_reasonable_yn` | y if the severity fits what the comment says |
| `issue_type_reasonable_yn` | y if the category fits |
| `grounded_in_diff_yn` | y if everything it claims is actually in the diff shown |
| `notes` | anything, optional |

Do not open `answer_key.csv` until you have finished your rows.
Then: `python -m prana.eval.human score --sheet ratings.csv`
