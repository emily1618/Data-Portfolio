# Increase Enrollment for the Program 

##### Author: Emi Ly

##### Date: Feb 28, 2022

### [Problem](#problem)
### [Dataset](#foc-dataset)
### [Data Cleaning](#foc-data-cleaning)
### [Analysis](#foc-analysis)
### [Dashboard](#foc-dashboard)


## PROBLEM

There are two ongoing problems for this Houston non-profit organization raising concern for all skate holders, including funders, the board, and the public:

➡ Decreasing enrollment for the Financial Opportunity Center, where clients come in to improve their financial means by attending the financial workshop and job training.

➡ Increasing need for Houston families to get utility assistance, but applications also take longer to process due to lack of staffing and stricter funder requirement.

The center's FOC is funded by a few major non-profit organizations in the Greater Houston area. One of them, United Way, is to create the opportunity for individuals and families in the Greater Houston community to thrive. United Way will focus on and invest in high-quality programs focused on serving ALICE (Asset Limited, Income Constrained, Employed) and those living below the Federal Poverty Level.  One of the programs, the Financial Opportunity Center (FOC), is a career and personal financial service center that focuses on the ALICE population's financial bottom line. FOC provides families with services across three areas:

1. Employment Placement and Career Improvement
2. Financial Education and Coaching
3. Public Benefits Access

For some time, our center's FOC has been struggling with decreasing enrollment, especially in 1 and 2 above. Under 3. Public Benefits Access, one of the primary sub-programs is CEAP, which helps pay 12 months of electricity and gas for lower-income families. With rising global temperature (especially in TX!) and increased energy costs to cool the homes, this is a necessary and popular benefit program in Houston.

We will explore the FOC problem in this folder. Please find the CEAP problem and analysis at: https://github.com/emily1618/Data-Portfolio/tree/main/CEAP


### Break down of FOC problem:
- Enrollment is counted when a client completes a zoom or in-person orientation, a soft credit check, and signed paperwork.
- There is a specific time frame, but generally, the center would like the client to complete all consultation, training, exit with a job, OR improve credit score within six months to one year.
- Decreasing enrollment started drastically falling since 2017. 2016 is high. 
- The other departments in the center also have been experiencing a decline in enrollment (e.g.: child care center), but not as steep of a decline as FOC. FOC also received the most funding out of all other departments. 
- The average age of FOC enrollment is the 40 group, and there is a slight decrease throughout the year. I need to check race demographic data later. 
- There are a few possible causes. One may be the center to outreaching to the wrong group of clients. Second, the quality and number of job training classes (need further analysis on the quality and outcome of the types), and maybe there is less of ALICE population around the center (need data to support this claim). 
- Metric is user enrollment and successful exit. The client defines a successful exit if improved their credit score, completing the job training, and landing a job relevant to the training. 
- Outreach idea to increase enrollment and launch the ideas. What is the cost of outreach and staffing requirements? How much of a percentage increase in enrollment do we want to see based on the budget?


#### What I completed:
✔ I shared the enrollment data insight via the Tableau dashboard to the stakeholders. I also recommend ideas to outreach to a specific population so marketing strategy can be more focused, producing more interest in enrollment.



#
## FOC DATASET

Enrollment data from 2012 to 2022

Accuracy: data is accurate because the center verified client information with their ID. For each training/consultation, the staff will need to record to be counted for. 

Relevancy: data is relevant because there are paperwork, document, and credit check to verify client information.

Completeness: data is not complete; there many missing data that need further investigation.

Timeliness: data is up to date.

Consistency: the data is consistent. Data is entered through the Salesforce platform. Therefore, the platform should have a consistent format.

🚫 Because the dataset contains real client names and information. This notebook is public hence some of the codes are commented out, so no df containing the client name will be printed. 



#
## FOC DATA CLEANING

The dataset is first cleaned in Excel including but not limited to the following:
- birthday data type
- birthday entered wrong (eg: 1/1/5674, 12-5/1968, 123/2/1970, Agust 5)
- birthday not completed
- using letter O instead of number 0 in dates
- spell out birthday month (eg: August, 30,)
- numerical currency data that's written out (eg: $300/month, 500 job + $400 ssi)
- currency having . and , in wrong places (eg: 6.66.7, 134,05)
- number stored as object or the number is stored as text (use `=VALUE()`) to convert back. Use `=IF(AK2="","",VALUE(AK2))` to convert to float number in excel if containing NaN in rows.
- extremely high monthly income
- negative birthday
- negative income
- find all and replace to fix spelling in the utility company columns to keep it consistent (eg: V24/7, v 247, V247 company)
- turn name into cap for first letter using `=PROPER`
- grab the year off birthday using `=TEXT(cell, "yyyy")`
- use `=LEFT()` or `=RIGHT()` to grab the specific number for household size-long
- use text to column to fix values
- `=DOLLAR()` to convert currency if needed

Using `describe()` in Python, I run into more issue with the numerical data so the dataset needs to be examine and clean further in Excel. 


#
## FOC ANALYSIS

### SQL
- Updating


### Python EDA Notebook:
- https://github.com/emily1618/Data-Portfolio/blob/main/CEAP/ceap.ipynb


#
## FOC DASHBOARD


#### Using Data To Increase Enrollment 🎨[Click Here For Tabealu Public Viz: Using Data To Increase FOC Enrollment](https://public.tableau.com/app/profile/emily.liang7497/viz/IncreaseMemberEnrollmentUsingData/Dashboard1?publish=yes)

![foc tableau 1](https://user-images.githubusercontent.com/62857660/155892915-05e6715a-b578-453e-8df6-2ff8829c2a94.jpg)![foc tableau 2](https://user-images.githubusercontent.com/62857660/155892917-2dcf8192-2a26-4cab-9231-8d9c3ccf6ecd.jpg)![foc tableau 3](https://user-images.githubusercontent.com/62857660/155892919-7eac5787-699b-44ae-b371-ddcaea1dd40b.jpg)





