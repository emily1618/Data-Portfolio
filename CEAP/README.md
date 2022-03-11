# Increase Funding for the Program 

##### Author: Emi Ly

##### Date: Feb 28, 2022

### [Problem](#problem)
### [Dataset](#ceap-dataset)
### [Data Cleaning](#ceap-data-cleaning)
### [Analysis](#ceap-analysis)
### [Dashboard and Presentation](#ceap-dashboard-and-presentation)


## PROBLEM

There are two ongoing problems for this Houston non-profit organization raising concern for all skate holders, including funders, the board, and the public:

➡ Decreasing enrollment for the Financial Opportunity Center, where clients come in to improve their financial means by attending the financial workshop and job training.

➡ Increasing need for Houston families to get utility assistance, but applications also take longer to process due to lack of staffing and stricter funder requirement.

The center's FOC is funded by a few major non-profit organizations in the Greater Houston area. One of them, United Way, is to create the opportunity for individuals and families in the Greater Houston community to thrive. United Way will focus on and invest in high-quality programs focused on serving ALICE (Asset Limited, Income Constrained, Employed) and those living below the Federal Poverty Level.  One of the programs, the Financial Opportunity Center (FOC), is a career and personal financial service center that focuses on the ALICE population's financial bottom line. FOC provides families with services across three areas:

1. Employment Placement and Career Improvement
2. Financial Education and Coaching
3. Public Benefits Access

For some time, our center's FOC has been struggling with decreasing enrollment, especially in 1 and 2 above. Under 3. Public Benefits Access, one of the primary sub-programs is CEAP, which helps pay 12 months of electricity and gas for lower-income families. With rising global temperature (especially in TX!) and increased energy costs to cool the homes, this is a necessary and popular benefit program in Houston.

In this folder, we will explore the CEAP program. We were facing problems such as:
- Not enough staff during the busy season to handle the number of clients and process the applications. 
- The organization also is using an outdated system that was in place 7+ years ago. 


#### Break down of CEAP problem:
- Metric is pay per application and staff hours needed to complete the application. 
- The funding is paid per application. Therefore, the more the center's application can complete, the higher the revenue. However, the payment is capped at a maximum of 2000 applications per year. Still, the potential to amend the contract for a higher amount is possible (the funder confirmed that they would like the center to make more applications). 
- If the center can process applications efficiently promptly, then the staff can outreach more to gain more clients for more applications for more pay per application
Measure success, increase pay per application, and reduce staff hours needed to complete the application or hire additional staff should processing time stay the same. 
- what is the cost and staffing requirement? How many applications do we need to complete to justify hiring an additional staff even with the rate increased? 

#### What I completed:
✔ I replaced the system from a 7+ years old paper system/storage into an electronic version, including online application and online scheduling. The data is stored in JotForm and entered automatically by the clients instead of a staff manually typing in every week into Access. In addition, the report can be downloaded and generated in minutes.

✔ Using data insight and visualization, I negotiated a new contract with the funder, increasing pay per application by **100%** for the utility assistance program. 

✔ I also secured **100%** funding from another source for two years without annual contract renewal (In the past, the budget had to be re-submitted an RFP and renewed annually based on performance).

#### What I will do:
- Using data to predict how many staffs are needed during the busiest months.

#
## CEAP DATASET

**CEAP DATASET: Application data from 2020 to 2021**

- Accuracy: data is accurate because the center verified client information with their ID/passport/perm card and proper income documents. 

- Relevancy: data is relevant because clients need to submit official documents pertinent to their application to apply for utility assistance.

- Completeness: there is minimal missing data, less than 2.5% in one couple of columns.

- Timeliness: data is up to date. Clients need to sign and date their applications.

- Consistency: the data is consistent after cleaning in excel and python. 

🚫 Because the dataset contains real client names and information. This notebook is public hence some of the codes are commented out, so no df containing the client name will be printed. The snippet of CEAP dataset:

![Capture](https://user-images.githubusercontent.com/62857660/156033093-aa8462b4-7eca-4aab-9460-2e4a98549c73.jpg)



#
## CEAP DATA CLEANING

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


#
## CEAP ANALYSIS

### SQL
- Updating


### Python EDA Notebook:
- https://github.com/xtenix88/Data-Portfolio/blob/main/Getting-Funding/ceap.ipynb


#
## CEAP DASHBOARD AND PRESENTATION

#### 🎞[Funding Proposal Video Presentation: Funding Proposal 2022](https://youtu.be/g8oMgNmdNMU)

![youtube](https://user-images.githubusercontent.com/62857660/157914623-20016f30-066c-4d44-a0f7-dac22052401f.JPG)


#### 🎨[Tabealu Public Viz: Using Data To Increase Funding]https://public.tableau.com/app/profile/emily1618/viz/UtilityApplicationDashboardClientData/Dashboard1

![dash](https://user-images.githubusercontent.com/62857660/157909225-af830b0e-2fb7-4681-9619-58bb572f31d8.JPG)




