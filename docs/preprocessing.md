### Step by step documentation for preprocessing.

- Remove unnecessary columns:
#### Job descriptions
```
Not needed for our case:
- Job ID
- Posting Type
- # Of Positions
- Title Code No
- Level
- Job Category: 0,1,2," "
- Full-Time/Part-Time indicator
- Division/Work Unit
- Additional Information
- To Apply
- Work Location 1
- Recruitment Contact
- Posting Date
- Post Until
- Posting Updated
- Process Date

Too hard to work with:
- Salary Range From
- Salary Range To
- Salary Frequency
- Work Location
- Hours/Shift
- Residency Requirement (for now, feature engineering needed)
```

#### Resume
```
Not needed for our case:
- ID
- Resume_html
```

- Remove duplicate rows
because, unnecessary and over fits the model on the same data. "thinks its more important, because it occurs more often"

- Remove HTML/XML tags:
because, other wise we get tokens like "b" from tags like (`<b>`). we dont want extra noise

- Remove websites or links
because, increases noise for the model and is not necessary for our case

- Only Alpha Numerical tokens
because, this removes special characters like '#, @, *' and punctuations

- Remove extra whitespaces
because, not necessary and increases noice  

- No stopwords removal
because, encoder models like all mini lm l6 v2 use self attention to get context out of the stopwords, if removed --> less context

- No stemming or lemmatization
because, encoder models are trained on full words, you reduce context by making 'running' --> 'run' 

- Keeping capitalization
because, some words have complete different meaning when put to lowercase, for example: "US" --> "us"

- Apply Tokenization (All mini lm l6 v2)


(imput missing rows with NO DESCRIPTION)
