
-- Two data table: Covid Deaths and Covid Vaccainations
Select *
FROM PortfolioProject..CovidDeaths
ORDER BY location, date

Select *
FROM PortfolioProject..CovidVaccinations
ORDER BY location, date

-- Data we need for the analysis
Select location, date, total_cases, new_cases, total_deaths, population
FROM PortfolioProject..CovidDeaths
ORDER BY Location, date

-- Total Cases vs Total Deaths 
-- Shows likelihood of dying if you get covid in United States
Select Location, date, total_cases, total_deaths, (Total_deaths/total_cases)*100 AS death_percentage
FROM PortfolioProject..CovidDeaths
WHERE location like '%states%'
ORDER BY Location, date

-- Shows likelihood of dying if you get covid in China
-- China actually has a higher precentage of death, but their total deaths of 4632 has not changed since 2020-05-16. US total deaths on the same date is 92073.
Select Location, date, total_cases, total_deaths, (Total_deaths/total_cases)*100 AS death_percentage
FROM PortfolioProject..CovidDeaths
WHERE location like '%china%'
ORDER BY Location, date

-- Total Cases vs Population
-- Shows percentage of population got Covid in United States
Select Location, date, total_cases, population, (total_cases/population)*100 AS population_percentage
FROM PortfolioProject..CovidDeaths
WHERE location like '%states%'
ORDER BY Location, date

-- Shows percentage of population got Covid in China
Select Location, date, total_cases, population, (total_cases/population)*100 AS population_percentage
FROM PortfolioProject..CovidDeaths
WHERE location like '%china%'
ORDER BY Location, date

-- Shows percentage of population got Covid for other countries
Select Location, date, total_cases, population, (total_cases/population)*100 AS population_percentage
FROM PortfolioProject..CovidDeaths
ORDER BY Location, date

-- Countries with highest infection rate
Select Location, population, MAX(total_cases) as highest_infection_count, MAX((total_cases/population))*100 AS percentage_population_infected
FROM PortfolioProject..CovidDeaths
GROUP BY Location, Population
ORDER BY percentage_population_infected desc

-- Countries with highest death rate per population
Select Location, population, MAX(cast(total_deaths as int)) as deaths, MAX((cast(total_deaths as int)/population))*100 AS percentage_population_died
FROM PortfolioProject..CovidDeaths
GROUP BY Location, Population
ORDER BY percentage_population_died desc

-- Continent with highest death toll
Select location, MAX(cast(total_deaths as int)) as total_deaths_count
FROM PortfolioProject..CovidDeaths
WHERE continent is null
GROUP BY location
ORDER BY total_deaths_count desc

-- Countries with highest death toll
Select Location, MAX(cast(total_deaths as int)) as total_deaths_count
FROM PortfolioProject..CovidDeaths
WHERE continent is not null
GROUP BY Location
ORDER BY total_deaths_count desc

-- Global percentage death
SELECT sum(total_cases) as sum_of_cases, sum(cast(total_deaths as int)) as sum_of_deaths, sum(cast(new_deaths as int))/sum(new_cases)*100 as global_death_percentage
FROM PortfolioProject..CovidDeaths
WHERE continent is not null


-- Total population vs vaccinations in rolling basis for countries. 
-- Issue encounter: vaccinated people are more than the population, say for Canada and United States. Could it be it count each dose as one vaccination?
SELECT deaths.continent, deaths.location, deaths.date, deaths.population, vacc.new_vaccinations, 
	SUM(CAST(vacc.new_vaccinations as int)) OVER (Partition by deaths.location ORDER BY deaths.location, deaths.date ) AS rolling_people_vaccinated
	-- (rolling_people_vaccinated/population)*100
FROM PortfolioProject..CovidDeaths deaths
JOIN PortfolioProject..CovidVaccinations vacc
 ON deaths.location = vacc.location
	AND deaths.date = vacc.date
WHERE deaths.continent is not null 
ORDER BY Location, date, population

-- CTE
WITH popVSvacc (continent, location, date, population, new_vaccinations, rolling_people_vaccinated) 
AS (
SELECT deaths.continent, deaths.location, deaths.date, deaths.population, vacc.new_vaccinations, 
	SUM(CAST(vacc.new_vaccinations as int)) OVER (Partition by deaths.location ORDER BY deaths.location, 
	deaths.date ) AS rolling_people_vaccinated
FROM PortfolioProject..CovidDeaths deaths
JOIN PortfolioProject..CovidVaccinations vacc
 ON deaths.location = vacc.location
	AND deaths.date = vacc.date
WHERE deaths.continent is not null 
)
SELECT *, (rolling_people_vaccinated/population)*100 as rolling_vacc_percentage
From popVSvacc


-- Temp Table
DROP TABLE IF EXISTS #percent_pop_vaccinated
CREATE TABLE #percent_pop_vaccinated
(
continent nvarchar(225),
location nvarchar(225),
date datetime,
population numeric,
new_vaccinations numeric,
rolling_people_vaccinated numeric
)
INSERT INTO #percent_pop_vaccinated
SELECT deaths.continent, deaths.location, deaths.date, deaths.population, vacc.new_vaccinations, 
	SUM(CAST(vacc.new_vaccinations as int)) OVER (Partition by deaths.location ORDER BY deaths.location, 
	deaths.date ) AS rolling_people_vaccinated
FROM PortfolioProject..CovidDeaths deaths
JOIN PortfolioProject..CovidVaccinations vacc
 ON deaths.location = vacc.location
	AND deaths.date = vacc.date
WHERE deaths.continent is not null 
SELECT*, (rolling_people_vaccinated/population)*100 AS rolling_vacc_percentage
FROM #percent_pop_vaccinated


-- Create View to store data for later visualization
CREATE VIEW percent_pop_vaccinated_global AS
SELECT deaths.continent, deaths.location, deaths.date, deaths.population, vacc.new_vaccinations, 
	SUM(CAST(vacc.new_vaccinations as int)) OVER (Partition by deaths.location ORDER BY deaths.location, 
	deaths.date ) AS rolling_people_vaccinated
	-- (rolling_people_vaccinated/population)*100
FROM PortfolioProject..CovidDeaths deaths
JOIN PortfolioProject..CovidVaccinations vacc
 ON deaths.location = vacc.location
	AND deaths.date = vacc.date
WHERE deaths.continent is not null 
	

SELECT *
FROM percent_pop_vaccinated_global
