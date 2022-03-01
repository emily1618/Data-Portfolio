# Increase Funding and Enrollment For a Houston Non-Profit

##### Author: Emi Ly

##### Date: Feb 28, 2022

### [Intro](#introduction)
### [Data](#dataset)
### [Excel](#excel-cleaning)
### [SQL](#sql-query)
### [Python](#python-eda)
### [Presentation](#presentation-viz)

#

## INTRODUCTION

To begin, there are two ongoing problems for this Houston non-profit organization raising concern for all skate holders including funders, the board, and the public:

➡ Decreasing enrollment for the Financial Opportunity Center, where clients come in to improve their financial means by attending financial workshop and job training.

➡ Increasing need for Houston families to get utility assistance, but applications also take longer time to process due lack of manpower and stricter funder requirement.

The center's FOC is funded by few major non-profit organizations in the Greater Houston area. One of them, United Way, is to create the opportunity for individuals and families in the Greater Houston community to thrive. United Way will focus on and invest in high-quality programs focused on serving ALICE (Asset Limited, Income Constrained, Employed) and those living below the Federal Poverty Level.  One of the programs, the Financial Opportunity Center (FOC) is a career and personal financial service center that focus on the financial bottom line for the ALICE population. FOC provides families with services across three areas:

1. Employment Placement and Career Improvement
2. Financial Education and Coaching
3. Public Benefits Access

For some times, our center's FOC has been struggling with decreasing enrollment, especially in 1 and 2 above. Under 3. Public Benefits Access, one of the primary sub-programs is CEAP, which helps pay 12 months of electricity and gas for lower income families. With global rising temperature (especially in TX!) and increased energy cost to cool the homes, this is necessary and popular benefit program in the Houston.

We were facing problems such as:
- Not enough staffs during the busy season to handle the number of clients and process the applications. 
- The organization also is using an outdated system that was in place 7+ years ago. 

### Break down of FOC problem:
- Enrollment is counted when a client successfully completed an zoom or in-person orientation, a soft credit check, and signed paperwork.
- There is specific time frame, but generally, the center would like the client to complete all consultation, training, exit with a job OR improving credit score within 6  months to one year.
- Decreasing enrollment started drastically falling since 2017. 2016 is the high. 
- The other departments in the center also have experiencing decline in enrollment (eg: child care center), but not as steep of a decline as FOC. FOC also received the most fundings out of all other departments. 
- The average age of FOC enrollment is the 40 group and there is a slight decrease in age throughout the year. Need to check race demographic data later. 
- There are few possible causes. One may be the center to outreaching to the wrong group of clients. Second, the quality and number of job taining classes (need further analysis on the quality and outcome of the classes), and maybe there is less of ALICE population around the center (need data to support this claim). 
- Metric is user enrollment and successful exit. Successful exit is defined by client improved their credit score, completion of the a job training, and landing a job relvant to the training. 
- Outreach idea to increase enrollment and launching the ideas. What are the cost of outreach and manpower requirement? How much of a percentage increase in enrollment we want to see based on the budget? 


### Break down of CEAP problem:
- Metric is pay per application and staff hours needed to complete application. 
- The funding is pay per application. The more the application the center can complete, the higher the revenue. The revenue is capped at a maximum of 2000 applications per year but potential to amend the contract for higher amount is very possible (confirmed by funder and they would like the center to do more applications). 
- If the center is able to process application efficiently in a timely manner, then the staff can outreach more to gain more client for more applications for more pay per application
- Measure of success, increase pay per application and reduce staff hours needed to complete application or hire additional staffs should processing time stay the same. 
- what are the cost and manpower requirement? How many applications we need to complete to justify hiring an additional staff even with the rate increased? Maybe using season data entry. 

### What I completed :
✔ I replaced the system from a 10 years old paper system/storage into electronic version including online application and online scheduling. The data is stored in the cloud and is updated automatically instead of manually typing in every week into Access. Report can be downloaded and generated under minutes.

✔ Using data insight and visualization, I negotiated an new contract with the funder, increasing pay per application by **100%** for the utility assistance program. 

✔ I also secured **100%** funding from another source for two years without annual contract renewal (which is typically require to submit RFP and renew each year based on performance).

### What I will do:
- Using data to predict how many staffs are needed during the busiest months.

- **Even though it's a non-profit, the center will still need to run like a for-profit business to sustain. Only way is to increase enrollment so the center can increase funding. More enrollment means the center is growing and able to help the Houston community efficiently. This  will allow the center to request for additional funding from existing/new funders. I will be exploring the data provided by this FOC from 2012 to 2022 to provide insights on how to increase enrollment.**

#
## DATASET

There will be **two** dataset. One is FOC client enrollment data from 2012 to 2022 and one is CEAP utility assistant client data from 2020 to 2021. Each dataset will be analyzed seperately. 

**FOC DATASET:**

- Accuracy: data is accurate because the center verified client information with their ID. For each training/consulation, the staff will need to make a record to be counted for. 

- Relevancy: data is revelant because there are paperwork,document, and credit check to verify client information.

- Completeness: data is not complete, there many missing data which need further investigation.

- Timeliness: data is up to date.

- Consistency:the data is consistent. Data is enter through Salesforce platform. The platform should have consistent format. 

**CEAP DATASET:**

- Accuracy: data is accurate because the center verified client information with their ID and proper income documents. For each application, the staff will need to make a record to be counted for credit. 

- Relevancy: data is revelant because there are paperwork, document, and credit check to verify client information for the intend use of giving them utility assistance.

- Completeness: there is minimal missing data, less than 2.5%.

- Timeliness: data is up to date.

- Consistency:the data is consistent after some format cleaning in excel and python. 

🚫 Because the dataset contains real client names and information. This notebook is public, hence some of the code are commented out so no df containing the client name will be printed. Snippet of CEAP dataset:
![Capture](https://user-images.githubusercontent.com/62857660/156033093-aa8462b4-7eca-4aab-9460-2e4a98549c73.jpg)



#
## EXCEL CLEANING

The dataset is first cleaned in Excel including but not limited to the following:
- birthday data type
- birthday entered wrong (eg: 1/1/5674, 12-5/1968, 123/2/1970)
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


## SQL QUERY

- Updating


## Python EDA

##### Increase Funding For CEAP notebook- https://github.com/xtenix88/Data-Portfolio/blob/main/Getting-Funding/ceap.ipynb
##### Increase Enrollment for FOC notebook- https://github.com/xtenix88/Data-Portfolio/blob/main/Getting-Funding/foc%20-%20ongoing.ipynb - Ongoing


## Presentation Viz

#### Using Data To Increase Enrollment 🎨[Click Here For Tabealu Public Viz: Using Data To Increase FOC Enrollment](https://public.tableau.com/app/profile/emily.liang7497/viz/IncreaseMemberEnrollmentUsingData/Dashboard1?publish=yes)

![foc tableau 1](https://user-images.githubusercontent.com/62857660/155892915-05e6715a-b578-453e-8df6-2ff8829c2a94.jpg)![foc tableau 2](https://user-images.githubusercontent.com/62857660/155892917-2dcf8192-2a26-4cab-9231-8d9c3ccf6ecd.jpg)![foc tableau 3](https://user-images.githubusercontent.com/62857660/155892919-7eac5787-699b-44ae-b371-ddcaea1dd40b.jpg)

#### Using Data to Increase Funding






