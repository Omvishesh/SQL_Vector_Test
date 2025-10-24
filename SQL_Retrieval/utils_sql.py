from   utils_common import llm_call, openai_call
from   textwrap import dedent
import re
import pandas as pd
import ast

def classify_query(query):
    system_instruction=dedent("""
                You are tasked with classifying the given query into one of the following categories: "CPI", "GDP", "IIP", "MSME", "agriculture_and_rural", "social_migration_and_households", "enterprise_surveys", "worker_surveys", "district_level" or "Out of domain". Use the guidelines below to analyze key entities and determine the best fit. Respond only with the selected category name—do not include reasoning, explanations, or additional text.
                If any query is asking for district level data, classify it into "district_level" class.

                You proceed using the following hints:

                # 1. Analyze the provided query for key entities.

                ## finance_and_industry

                a. Classify queries related to insurance, irdai, insurers, non-life insurers, financial markets, Foreign Direct Investment (FDI), e-shram, investments, revenues, co2 emissions, electric vehicles(ev), oil and gas reserves, commodities export and import, international trade,port trade energy/commodities, renewable energy capacity, mutual funds, data of toll collection, NETC (National Electronic Toll Collection) transactions,  financial data for Assam, petroleum/renewables, or state/sector indicators not directly in CPI, GDP, IIP, MSME, agriculture, or social, indian companies data (opened, closed registered) and companies compliance.

                b. The following files comprise the finance_and_industry datasets:[irdai_nonlife_india_mth_insurer, fpi_india_yr_invtype, wages_sector_industry_index, eshram_state_dly_registrations, stock_india_mth_boaccounts, stock_india_mth_dps, fdi_india_qtr_state, revenue_maharashtra_fy_category, mf_monthly_schemes, statewise_petroleum_consumption,
                co2_emissions_by_fuel_yearly, quick_estimates_major_commodities_july_export, quick_estimates_major_commodities_july_import, statewise_cumulative_renewable_power, marketcap_nse_india_mth, ki_assam_mth_sctg, insurance_india_mth_sctg, toll_state_monthly_etc_transactions, ev_state_yr_catg, trade_india_mth_commodity_import, trade_india_mth_commodity_export, trade_india_mth_stateut_country_export,
                trade_india_mth_stateut_country_export, trade_india_mth_commoditygrp_country_export, fdi_india_fy_state, fdi_india_fy_sector, fdi_india_fy_country, energy_india_statewise_crude_oil_ngas_reserves, energy_india_statewise_coal_reserves, crude_oil_mth_data, ppac_mth_petroleum_consumption, ores_minerals_exports_yearly mf_india_qtr_total, insurance_india_mth_life_insurer, trade_india_annual_country,
                trade_india_mth_region_commodity, port_dwell_time_month, trade_india_mth_country, trade_india_mth_region, cumulative_capacity_state_month, netc_india_mth_stats, ins_nat_yr_fin_highlights, company_india_annual_type_mca, energy_nat_ann_petroleum_products_consumption, energy_wld_nat_mth_crude_petroleum_trade, energy_nat_mth_crude_oil_prod, energy_nat_mth_petroleum_prod, mca_nat_mth_comp_closed, mca_nat_mth_companies_registered].

                ## CPI

                a. Classify queries related to inflation, CPI, price indices, Monthly Per Capita Expenditure (MPCE), sales, wholesale, consumer, or consumption. This includes datasets on general CPI inflation, agricultural and rural laborer price indices, city-wise housing prices, wholesale price indices (financial year and calendar wise), and worker-specific CPI data.

                b. The following files comprise the CPI datasets: [cpi_state_mth_grp_view, cpi_state_mth_subgrp_view, cpi_india_mth_grp_view,cpi_india_mth_subgrp_view,consumer_price_index_CPI_for_agricultural_and_rural_labourers,city_wise_housing_price_indices,whole_sale_price_index_WPI_financial_year_wise,cpi_worker_data,whole_sale_price_index_WPI_calendar_wise, cpi_iw_point_to_point_inflation, cpi_iw_centre_index, cpi_iw_retail_price_index, hces_india_yr_sector].
                Any query that can be answered with these data sets should be classified as "CPI".

                ## GDP

                a. Classify queries on GDP, gross domestic product, GSDP, GVA (Gross Value Added), NDP, NNI, GNI, capital formation, expenditure components, per capita values, NSDP, NSVA, PCNSDP, state economic indicators, or macro-level metrics like monetary policy instruments (repo rate, bank rate, MSF, SLR, CRR), government securities yields, call money rates, forward premia, balance of payments, external debt, foreign reserves, FDI, portfolio investments, ECB, trade balance, exports, imports, money supply (M1, M3), bank credit, government borrowings, fiscal deficit, market capitalization, and exchange rates. This covers datasets on annual GDP estimates (crore and growth rates), gross state values, national account aggregates, per capita income and consumption, provisional GDP macro aggregates, quarterly expenditure and GDP estimates, other macro indicators (daily, monthly, quarterly, weekly), top macro indicators (monthly, quarterly, weekly), state-wise NSDP/NSVA/PCNSDP, IMF exports.
                ** Notes **
                All queries about "CREDIT" should go to MSME and must not go to GDP

                b. The following files comprise the GDP datasets: [annual_estimate_gdp_growth_rate,gross_state_value,key_aggregates_of_national_accounts,per_capita_income_product_final_consumption,provisional_estimateso_gdp_macro_economic_aggregates,quaterly_estimates_of_expenditure_components_gdp,quaterly_estimates_of_gdp , other_macro_economic_indicators_daily_data ,other_macro_economic_indicators_monthly_data,other_macro_economic_indicators_quaterly_data,other_macro_economic_indicators_weekly_data, top_fifty_macro_economic_indicators_monthly_data,top_fifty_macro_economic_indicators_quaterly_data,top_fifty_macro_economic_indicators_weekly_data , statewise_nsdp,statewise_nsva,statewise_pcnsdp, niryat_ite_commodity, niryat_ite_state].
                Any query that can be answered with these data sets should be classified as "GDP".

                ## IIP

                a. Entities such as IIP, industrial output, industrial production, material like cement, mining, manufacturing, electricity, pm schemes like pmay, pmgsy, motor vehicles or other industries should be classified as "IIP".

                b. The following files comprise the IIP datasets: [iip_india_yr_catg_view,iip_india_mth_catg_view,construct_state_cement_indicators_view, iip_india_yr_subcatg_view, iip_india_mth_subcatg_view, iip_in_assam, annual_chemical_production_data, vehicle_registrations_state].
                Any query that can be answered with these data sets should be classified as "IIP".

                ## MSME

                a. Classify queries on MSME, Micro, Small and Medium Enterprises, credit growth, exchange rates, Nifty SME, sambandh, procurement data, food/non-food credit,RBI, Payment Systems, Transactions,  gross bank credit,  payment systems in India, upi, upi transactions,RTGS (Real Time Gross Settlement), NEFT transaction, mobile banking statistics, internet banking transactions, regional/sectoral MSME distribution, or economic shares. This covers datasets on gross bank credit for food/non-food views, sector definitions, non-food credit details, regional and sectoral shares, industry views, priority sector views, daily Nifty SME index values, and state-wise Udyam registrations.

                b. The following files comprise the MSME datasets: [msme_gbc_food_non_food_view, msme_definitions_by_sector, msme_state_ureg_recent, msme_gbc_non_food_dtl_view, nifty_sme_index_daily_values , msme_share_by_region_view, msme_share_by_sector_view, msme_priority_sector_view, msme_industry_view, msme_global_view, upi_dly_stats, upi_mth_stats, upi_mth_failures, msme_sambandh_procurement_data, rbi_india_mth_payment_system_indicators, rbi_india_mth_bank_rtgs, rbi_india_mth_bank_neft, rbi_india_mth_bank_mobile_banking, rbi_india_mth_bank_internet_banking].
                Any query that can be answered with these data sets should be classified as "MSME".

                ## agriculture_and_rural

                a. Classify queries seeking descriptive statistics on agricultural households, including crop sales, farming inputs, irrigation, receipts, crop insurance (uptake, reasons for non-insurance), farmer advisory, MSP awareness, input procurement, land leasing/ownership, livestock, household classification, credit sources, asset investment, operational holdings, production/yield, annual temperature, fish production, watersheds, river basin catchment, coastline, annual rainfall data, faunal diversity, and social/regional comparisons. This aligns with themes like agricultural extension, insurance design, rural finance, food security, market linkages, land reform, asset creation, productivity, equity, and development. Datasets include distributions on crop sales by agency, farming resource use, seed quality/procurement, expenditure/receipts on assets and production, crop insurance experiences, land leasing by social group, operational holdings, livestock ownership, and more.

                b. The following files comprise the "agriculture_and_rural" datasets:[sa_agri_hhs_crop_sale_quantity_by_agency_major_disposal, sa_agri_hhs_reporting_use_of_diff_farming_resources, sa_agri_hhs_use_purchased_seed_by_quality, sa_avg_expenditure_and_receipts_on_farm_and_nonfarm_assets, sa_avg_gross_cropped_area_value_quantity_crop_production, sa_avg_monthly_expenses_and_receipts_for_crop_production, sa_avg_monthly_total_expenses_crop_production, sa_avg_monthly_total_expenses_receipts_animal_farming_30_days, sa_dist_agri_hh_not_insuring_crop_by_reason_for_selected_crop, sa_dist_agri_hhs_seed_use_by_agency_of_procurement, sa_dist_hhs_leasing_out_land_and_avg_area_social_group, sa_dist_of_agri_hhs_reporting_use_of_purchased_seed, sa_dist_of_hhs_by_hh_classification_for_diff_classes_of_land, sa_distribution_hhs_leasing_in_land_avg_area_social_group,
                sa_distribution_loan_outstanding_by_source_of_loan_taken, sa_distribution_operational_holdings_by_possession_type, sa_est_num_of_hhs_for_each_size_class_of_land_possessed, sa_estimated_no_of_hhs_for_different_social_groups, sa_no_of_hhs_owning_of_livestock_of_different_types, sa_no_per_1000_distri_of_agri_hhs_reporting_sale_of_crops, sa_no_per_hh_operational_holding_by_size_hh_oper_holding, sa_per_1000_agri_hh_insured_experienced_crop_loss, sa_per_1000_crop_producing_hh_crop_disposal_agency_sale_satisf, sa_perc_dist_of_land_for_hhs_belonging_operational_holding, sa_percent_distribution_of_leased_out_land_by_terms_of_lease, annual_mean_temperature, fish_production_yearly, watersheds_in_india, river_basin_catchment, coastline_population_and_length, rainfall_annualy_and_monthly, faunal_diversity]
                Any query that can be answered with these data sets should be classified as "agriculture_and_rural".

                ## social_migration_and_households

                a. Classify queries on household socio-economic patterns, migration (reasons, remittances, rural-urban shifts, income changes), finance sources, household ownership, asset ownership (housing, TV, cooler, AC), state-wise sanitation/water access, demographic statistics, river water quality statistics, digital connectivity (mobile, SIM, broadband, mass media), living standards (pucca housing, transport), public health/services,airport data, ayushman schemes, credit, aadhar data, hospital data, and equity (rural/urban, caste, gender). Datasets cover access to drinking water, labour wages, mass media/broadband, transport (air, water, land), public facilities, finance sources, latrine/handwashing, household assets, migration reasons/income changes, air conditioner/cooler possession, mobile usage, and residence changes.

                b. The following files comprise the "social_migration_and_households" datasets: [mis_access_to_improved_source_of_drinking_water, mis_access_to_mass_media_and_broadband, mis_availability_of_basic_transport_and_public_facility, mis_different_source_of_finance,
                    mis_exclusive_access_to_improved_latrine, mis_household_assets, mis_improved_latrine_and_hand_wash_facility_in_households, mis_improved_source_of_drinking_water_within_household,
                    mis_income_change_due_to_migration, mis_main_reason_for_leaving_last_usual_place_of_residence, mis_main_reason_for_migration, mis_possession_of_air_conditioner_and_air_cooler,
                    mis_usage_of_mobile_phone, mis_usual_place_of_residence_different_from_current_place, aadhaar_demographic_monthly_data, aadhaar_biometric_monthly_data, cghs_approved_hospital_data, labour_india_sector_industry_occupation_wages, labour_india_rural_wages,
                    airport_sewa_services_data, traffic_india_mth_air_passengers, sp_india_daily_state], demography_india_yr_popsexgrowth, demography_india_state_yr_literacy, hces_state_yr_assets, tlm_state_yr_transport_access, env_state_yr_river_water_quality]
                Any query that can be answered with these data sets should be classified as "social_migration_and_households".
                
                ## enterprise_surveys
                
                a. Classify queries related to enterprise-level characteristics and performance — including ownership type, registration status, nature of operation (perennial/seasonal), industry type, location (rural/urban), use of digital infrastructure (computers/internet), franchisee or NPI status, financial structure (capital invested, outstanding loans, working capital, banking access), and production metrics such as total output, intermediate consumption, and Gross Value Added (GVA) per enterprise. This class covers statistical surveys like ASI (Annual Survey of Industries) and NSS (National Sample Survey) enterprise modules that report on enterprise count, ownership pattern, economic activity, turnover, and profitability across industries and states. Queries under this class may involve enterprise distributions, average size, industrial classification (NIC codes), or productivity indicators at the enterprise level.

                b. The following files comprise the "enterprise_surveys" datasets: [asuse_est_annual_gva_per_establishment, asuse_est_num_establishments_pursuing_mixed_activity, asuse_per1000_estb_by_hours_worked_per_day, asuse_per1000_estb_by_months_operated_last_365days, asuse_per1000_estb_registered_under_acts_authorities, asuse_per1000_estb_using_computer_internet_last365_days, asuse_per1000_of_estb_using_internet_by_type_of_its_use, asuse_per1000_proppartn_estb_by_edu_owner_mjr_partner, asuse_per1000_proppartn_estb_by_other_econ_activities, 
                asuse_per1000_proppartn_estb_by_socialgroup_owner, asuse_per_1000_distri_of_establishments_by_nature_of_operation, asuse_per_1000_distri_of_establishments_by_type_of_location, asuse_per_1000_distri_of_establishments_by_type_of_ownership, asuse_per_1000_of_establishments_which_are_npis_and_non_npis, asuse_statewise_est_num_of_estb_pursuing_mixed_activity, asuse_statewise_est_num_of_estb_serving_as_franchisee_outlet, asuse_statewise_estimated_annual_gva_per_establishment_rupees, 
                asuse_statewise_per1000_distri_of_estb_by_nature_of_operation, asuse_statewise_per1000_distri_of_estb_by_type_of_location, asuse_statewise_per1000_distri_of_estb_by_type_of_ownership, asuse_statewise_per1000_estb_by_hours_worked_per_day, asuse_statewise_per1000_estb_by_month_num_operated_last365_day, asuse_statewise_per1000_estb_maintain_post_bank_saving_acc, asuse_statewise_per1000_estb_registered_diff_acts_authorities, asuse_statewise_per1000_estb_use_computer_internet_last365_day, 
                asuse_statewise_per1000_proppart_estb_by_social_grp_mjr_prtner, annual_survey_of_industries, asi_state_principal_characteristics, asi_imp_principal_characteristics_by_rural_urban_sector, asi_imp_principal_characteristics_india_by_mjr_indus_grp, asi_industrywise_factories_2022_23, asi_num_of_factories_nva, asi_statewise_number_of_factories_for_2022_23, asi_top_ten_states_by_number_of_factories, asi_trend_imp_characteristics_technical_coefficients, asi_trend_of_imp_characteristics_structural_ratios, 
                asi_trend_of_imp_principal_characteristics_india]
                
                Any query that can be answered with these data sets should be classified as "enterprise_surveys".

                ## worker_surveys
                
                a.Classify queries related to worker-level and employment characteristics — including worker distribution by gender, social group, education, employment level and employment  type (self-employed, regular wage, casual labor); working hours/days; wages, emoluments, and compensation; labor participation indicators such as LFPR (Labour Force Participation Rate), Gross Value Added (GVA) per worker, WPR (Worker Population Ratio), EPFO, and UR (Unemployment Rate). This class includes insights from PLFS (Periodic Labour Force Survey), NSS workforce modules, and establishment-based labor surveys. Queries may involve the number or proportion of workers across industries, average emoluments per worker, gender-wise employment trends, occupational classifications, or worker distribution by enterprise size and activity. 
                
                b. The following files comprise the "worker_surveys" datasets: [asuse_est_annual_emoluments_per_hired_worker,asuse_est_num_workers_by_employment_gender, asuse_est_value_key_characteristics_by_workers, asuse_estimated_annual_gva_per_worker_rupees, asuse_estimated_number_of_workers_by_type_of_workers, asuse_statewise_est_num_of_worker_by_employment_and_gender, asuse_statewise_estimated_annual_emoluments_per_hired_worker, aasuse_statewise_estimated_annual_gva_per_worker_rupees,
                asuse_statewise_estimated_number_of_workers_by_type_of_workers, asi_no_of_workers_and_person_engaged,periodic_labour_force_survey, lpfr_state_age, cws_industry_distribution_state, wpr_state_age, ur_state_age, epfo_india_mth_payroll, epfo_nat_ind_exempted_establishments_list]
                
                Any query that can be answered with these data sets should be classified as "worker_surveys".

                ## GST

                a. Classify queries on GST (including taxpayers, returns, state contributions, gross/net tax collections, IGST settlements,e-way bills, registrations, subsidies), GSTR filings, tax collections and refunds, and GST registrations.

                b. The following files comprise the GST datasets: [gst_registrations, gst_statewise_tax_collection_refund_data, gst_statewise_tax_collection_data, gst_settlement_of_igst_to_states, gstr_three_b, gstr_one, gross_and_net_tax_collection, gst_statewise_fiscal_year_collection_view,
                gst_statewise_fiscal_year_igst_settlement_view, gst_statewise_fiscal_year_refund_view, ewb_state_mth_stats].
                Any query that can be answered with these data sets should be classified as "GST".
                
                ## district_level
                
                a. Classify queries related to youth empowerment, employment, education, skill development indicators and any other query at the district level across India. These include queries about youth opportunity, opportunity metrics around population, csr spend, gdp growth, msme establishments etc, workforce participation, education and readiness scores, schools and college data, MSME distribution, skill training statistics, and other metrics that evaluate district-level youth development or performance, it also includes district level daily average rainfall data.
                
                b. The following file comprises the Youth Power dataset: [youthpower_district_level_metrics, rainfall_data]
                 Any query that can be answered using this dataset — such as those involving district-wise or state-wise youth power scores, unemployment rates, education indices, workforce data, MSME counts, or skill training statistics, district wise rainfall data — should be classified as "district_level".

            # 2. Rules for rejection as "Out of Domain"

            Any query that belongs to none of [finance_and_industry, CPI, GDP, IIP, MSME, agriculture_and_rural, social_migration_and_households, enterprise_surveys, worker_surveys, district_level] should be classified "Out of domain". Examples of out of domain queries include those about general queries about the economy, queries about government policies, queries about upcoming challenges, and queries unrelated to finance. All of these should be marked "Out of domain".

            # 3. EXTREMELY IMPORTANT:
                - Based on the above description, respond ONLY with one of the classes from the following list:
                    [CPI, GDP, IIP, MSME, GST, agriculture_and_rural, social_migration_and_households, enterprise_surveys, worker_surveys, finance_and_industry, district_level, Out of domain]
                - DO NOT include any reasoning traces or other text apart from the class selected from the above list.
            """)
    query_class, i_tokens, o_tokens = llm_call(system_instruction, query)
    return query_class.strip(), i_tokens, o_tokens

def file_selector_finance_and_industry(query):
    system_instruction = dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        # Table data
        1. eshram_state_dly_registrations : This table contains daily registration data for the e-Shram portal, broken down by state and district, showing counts by registration channel and date.
        Instructions: Use this table to analyze e-Shram registrations by state, district, channel (ssk, csc, self, umang, other_scheme), date, and fiscal year. Useful for tracking registration trends and comparing channels or regions.
        Example Queries:
        Query: Show total registrations for each district in ANDAMAN AND NICOBAR ISLANDS for fiscal year 2025-26.
        Query: Get the number of self registrations in South Andamans on 2025-09-24.
        Query: List total registrations by channel for each district on 2025-09-24.


        2. fpi_india_yr_invtype: This table contains annual data on Foreign Portfolio Investment (FPI) inflows into India, broken down by investment type (such as equity, debt, hybrid, mutual funds, and AIF) for each financial year.
        Instructions: Use this table to analyze FPI inflows into India by year and by different investment categories. You can filter by financial year, sum or compare investment types, and track cumulative totals over time.
        Example queries:
        Queries: Show total FPI inflows for each financial year.
        Queries: List the equity and debt inflows for the financial year 1994-95.
        Queries: Get the cumulative FPI inflows up to each year.
        Queries: Find the years where mutual funds equity inflows were greater than zero.


        3. wages_sector_industry_index: This table contains wage rate index (WRI) data by sector and industry, including base year, year, period, and index values.
        Instructions: Use this table to retrieve wage rate index information for specific sectors, industries, years, or periods. Filter by columns like sector, industry, year, or period_as_on to get relevant WRI data.
        Example Queries:
        Queries: Show the wage rate index for the Sugar industry in 2023.
        Queries: List all industries in the Manufacturing Sector with their latest wage rate index.
        Queries: Get the wage rate index for Oils & Fats industry as of 1st January 2023.


        4. irdai_nonlife_india_mth_insurer: This table provides monthly and cumulative premium data, market share, and growth percentages for non-life insurance companies in India, categorized by insurer and month.
        Instructions: Use this table to analyze premium collections, market share, and growth trends for non-life insurers in India by month, insurer, or category.
        Example Queries:
        Queries: Show the total premium collected by each insurer in July 2025.
        Queries: List insurers with negative growth in July 2025.
        Queries: What is the market share of Bajaj Allianz General Insurance Company Limited in July 2025?


        5. stock_india_mth_boaccounts: This table contains monthly data on the number of demat accounts in India, categorized by Banks, Custodians, and Stockbrokers, including new accounts opened, accounts closed, and total accounts at month-end.
        Instructions: Use this table to analyze trends in demat account openings, closures, and totals across different categories and months. Filter by year, month, or category as needed.
        Examples Queries:
        Queries: Show the total number of new accounts opened by Stockbrokers in August 2025.
        Queries: List the number of accounts closed by each category in August 2025.
        Queries: Get the total accounts at the end of August 2025 for all categories.
        Queries: Show the monthly trend of new accounts opened by Banks in 2025.


        6. stock_india_mth_dps: This table contains monthly data on the number of participants in different categories (such as Banks, Custodians, Stockbrokers) in India, including counts at the beginning and end of each month, as well as new registrations and cancellations.
        Instructions: Use this table to analyze trends in participant numbers by category, year, and month, or to track registrations and cancellations over time.
        Examples Queries:
        Queries: Show the number of stockbrokers at the end of each month in 2025.
        Queries: How many new banks were registered in July 2025?
        Queries: List the total participants at the beginning of July 2025 for all categories.


        7. fdi_state_qtr_view: This table provides quarterly Foreign Direct Investment (FDI) statistics for Indian states, including investment amounts in INR crores and USD millions, and the percentage share of each state.
        Instructions: Use this table to analyze or retrieve FDI data by state and quarter, including total investment amounts and percentage shares.
        Examples Queries:
        Queries: Show the FDI in USD million for Maharashtra in the quarter starting April 2025.
        Queries: List all states with their FDI percent for the quarter ending June 2025.
        Queries: Get the total FDI in INR crore for Gujarat across all quarters.


        8. revenue_maharashtra_fy_category: This table provides fiscal year-wise revenue data for Maharashtra, categorized by revenue type and sub-category, including revenue amounts, their percentage contribution to the total, and descriptions.
        Instructions: Use this table to analyze Maharashtra's revenue collection by fiscal year, category, and sub-category. You can filter by fiscal_year, category, or sub_category to get specific revenue figures and their share in total revenue.
        Examples Queries:
        Queries: Show total revenue for Maharashtra in 2022-23 by category.
        Queries: List all sub-categories under category A for 2022-23 with their revenue and description.
        Queries: What was the percent contribution of 'Union Excise Duties' in 2022-23?


        9. mf_monthly_schemes: This table contains monthly aggregated data on mutual fund schemes in India, including scheme categories, types, names, asset and folio counts, fund flows, AUM, and other key metrics, reported by the Association of Mutual Funds in India.
        Instructions: Use this table to analyze mutual fund scheme performance, inflows/outflows, assets under management, and scheme distribution by category, type, or time period. Filter by month, year, scheme type, or category to get specific insights.
        Examples Queries:
        Queries: Show the net inflow/outflow for all Income/Debt Oriented Schemes in August 2025.
        Queries: List the number of folios and net assets under management for each scheme in the latest available month.
        Queries: Get total funds mobilized and repurchased for Open ended Schemes in fiscal year 2025-26.


        10. statewise_petroleum_consumption: This table provides annual petroleum consumption data (in thousand tonnes) for each Indian state and union territory.
        Instructions: Use this table to analyze or retrieve petroleum consumption figures by state and fiscal year.
        Examples Queries:
        Queries: Show the petroleum consumption for each state in 2023-24.
        Queries: Which state had the highest petroleum consumption in 2023-24?
        Queries: List petroleum consumption for ANDHRA PRADESH across all years.


        11. co2_emissions_by_fuel_yearly: This table contains yearly CO2 emissions data in million tonnes (Mt) from coal and oil/gas sources, along with the year and last update date.
        Instructions: Use this table to analyze or retrieve annual CO2 emissions from coal and oil/gas, compare trends over years, or find the latest emission values.
        Examples Queries:
        Queries: Show total CO2 emissions from coal for each year.
        Queries: What was the oil and gas CO2 emission in 2010-11?
        Queries: List all years with their total CO2 emissions (coal + oil/gas).
        Queries: Which year had the highest coal CO2 emissions?


        12. quick_estimates_major_commodities_july_export: This table provides quick export estimates for major commodities, showing export values (in INR crore) for July and April-July periods across two consecutive years, along with percentage changes.
        Instructions: Use this table to analyze export performance, compare year-on-year changes for July or April-July periods, and identify trends in major commodity exports.
        Examples Queries:
        Queries: Show the export value and percentage change for Coffee in July 2025 compared to July 2024.
        Queries: List all commodities with more than 20% growth in exports in April-July 2025 compared to April-July 2024.
        Queries: What is the total export value for all commodities in July 2025?


        13. quick_estimates_major_commodities_july_import: This table provides quick estimates of major commodity imports, showing values in INR crore for July and April-July periods across two years, along with percentage changes.
        Instructions: Use this table to analyze import values and percentage changes for major commodities between July and April-July periods of consecutive years.
        Examples Queries:
        Queries: Show the percentage change in import value for all commodities in July 2025 compared to July 2024.
        Queries: List the import values for Vegetable Oil for both July 2024 and July 2025.
        Queries: Which commodities had a decrease in import value from April-July 2024 to April-July 2025?


        14. statewise_cumulative_renewable_power: This table provides state-wise annual cumulative renewable power capacity data in India, including breakdowns by source (small hydro, wind, bio power, waste to energy, solar), total capacity, and growth rate.
        Instructions: Use this table to analyze renewable power capacity by state/UT, year, and energy source. You can filter by state, year, or energy type, and aggregate or compare data across years or regions.
        Examples Queries:
        Queries: Show the total renewable power capacity for all states in 2024.
        Queries: List the growth rate percent of renewable power for Andhra Pradesh over the years.
        Queries: Which state had the highest solar capacity in 2023?
        Queries: Get the total wind power capacity for each state in fiscal year 2023-2024.
        Queries: Show all data for Arunachal Pradesh in 2023.


        15. marketcap_nse_india_mth: This table contains monthly market capitalization data (in lakhs) for the NSE India, organized by fiscal year and month.
        Instructions: Use this table to analyze or retrieve monthly market capitalization figures for NSE India, filtered by fiscal year, month, or date as needed.
        Examples Queries:
        Queries: Show the NSE market capitalization for each month in fiscal year 2025-26.
        Queries: Get the latest updated NSE market cap value.
        Queries: List all fiscal years available in the table.


        16. ki_assam_mth_sctg: This table contains monthly financial data for Assam, including budget estimates, actuals, and percentage comparisons for various revenue and expenditure indicators.
        Instructions: Use this table to analyze Assam's monthly budget performance, compare actuals to budget estimates, and review trends across different revenue and expenditure categories.
        Examples Queries:
        Queries: Show the actuals and budget estimates for all Revenue Receipts indicators.
        Queries: List the percentage of actuals to budget estimates for the current year for each indicator.
        Queries: Find the actuals for State Goods and Services Tax (SGST).


        17. insurance_india_mth_sctg: This table contains monthly sector-wise premium and related financial data for various insurance companies in India, including breakdowns by insurance type (fire, marine, motor, health, etc.), total premiums, growth, and market share.
        Instructions: Use this table to analyze or compare insurance companies' performance by sector, track premium growth, or view detailed breakdowns of insurance business lines for a given period.
        Examples Queries:
        Queries: Show the total premium collected by each insurer for the latest month.
        Queries: Compare the motor insurance premium of Acko General Insurance Ltd with the previous year.
        Queries: List insurers with health premiums above 1000.
        Queries: Get the total accretion for all insurers except the previous year.
        Queries: Show the breakdown of marine insurance (cargo and hull) for Bajaj Allianz General Insurance Co Ltd.

        18. toll_state_monthly_etc_transactions: This table contains monthly ETC (Electronic Toll Collection) transaction data for toll plazas, including fiscal year, month, plaza name, state, transaction count, and transaction amount.
        Instructions: Use this table to analyze monthly ETC transactions by toll plaza, state, or fiscal year, and to aggregate transaction counts or amounts over time or by location.
        Example Queries:
        Queries: Show the total transaction amount for each state in fiscal year 2025-26.
        Queries: List the monthly transaction counts for Bharthana Toll Plaza in Gujarat.
        Queries: Get the total number of transactions for each month across all plazas.

        19. ev_state_yr_catg: This table contains yearly data on electric vehicle (EV) registrations by state and vehicle category in India, including the number of EVs registered and their percentage share among total vehicle registrations.
        Instructions: Use this table to analyze EV registration trends across different states, years, and vehicle categories, or to compare the share of EVs in total vehicle registrations.
        Example Queries :
        Query: Show the total number of EVs registered in Maharashtra for each year.
        Query: List the top 5 states with the highest percentage share of EVs in total vehicle registrations for the year 2022-23.
        Query: Find the number of 2 Wheeler EVs registered in Karnataka in 2021-22.

        20. trade_india_mth_commodity_import: This table contains monthly import data for various commodities in India, including HS codes, commodity names, month, year, financial year, import values (in crores), percentage growth, and last update date.
        Instructions: Use this table to analyze or retrieve India's monthly commodity import statistics, such as import values, growth rates, or trends by HS code, commodity, month, or year.
        Examples Queries :
        Query: Show the total import value for IRON AND STEEL in 2025.
        Query: List all commodities with negative percentage growth in June 2025.
        Query: Get the import value and growth for each commodity in financial year 2025-26.

        21. trade_india_mth_commodity_export: This table contains monthly export data from India by commodity, including HS code, commodity name, month, year, financial year, export values (in crores), percentage growth, and last update date.
        Instructions: Use this table to analyze or retrieve India's monthly export statistics by commodity, HS code, time period, or growth trends.
        Examples Queries
        Query: Show the export value and growth for 'GLASS AND GLASSWARE.' in April 2025.
        Query: List all commodities exported in the financial year 2024-25 with their total export values.
        Query: Get the percentage growth for each commodity in March 2025.
        Query: Find the latest updated export data for HS code '39.00'.

        22. trade_india_mth_stateut_country_export: This table provides monthly export statistics for Indian states and union territories, including total exports, exports for March and February, monthly growth percentage, and share in total exports for each fiscal year.
        Instructions: Use this table to analyze and compare export performance of Indian states/UTs by month, fiscal year, or growth trends. Filter by 'states_ut_name' for specific regions, or by 'fiscal_year' for year-wise data.
        Example Queries :
        Query: Show the top 3 states by total exports for fiscal year 2024-25.
        Query: List states with more than 10% growth in exports compared to the previous month for 2024-25.
        Query: Get March export values for all states for the latest fiscal year.

        23. trade_india_mth_commoditygrp_country_export: This table provides monthly export data of India by commodity group, including total exports, exports for March and February (in million dollars), percentage growth compared to the previous month, percentage share in total exports, and fiscal year.
        Instructions: Use this table to analyze India's export performance by commodity group for specific months, compare monthly export values, calculate growth rates, or determine the share of each commodity group in total exports for a given fiscal year.
        Example Queries :
        Query: Show the March 2025 export values for all commodity groups.
        Query: Which commodity group had the highest percentage growth in exports compared to the previous month in fiscal year 2024-25?
        Query: List the total exports and percentage share for Engineering Goods in the latest fiscal year.
        Query: Find the commodity groups where exports decreased in March compared to February for 2024-25.

        24. fdi_india_fy_state: This table contains data on Foreign Direct Investment (FDI) inflows into Indian states by financial year, including FDI amounts in INR crore and USD million, inflow percentage, and state names.
        Instructions: Use this table to analyze FDI inflows by state and financial year, compare FDI amounts across states, or examine trends in FDI distribution.
        Example Queries :
        Query: Show the top 5 states by FDI inflow in USD for the financial year 2024-25.
        Query: List the FDI inflow percentage for each state in 2024-25.
        Query: What was the total FDI in INR crore received by Karnataka over all years?

        25. fdi_india_fy_sector: This table contains data on Foreign Direct Investment (FDI) inflows into India by sector and financial year, including values in INR crore, USD million, and percentage share.
        Instructions: Use this table to analyze FDI inflows by sector, compare sector-wise FDI amounts, or examine trends across financial years.
        Example Queries :
        Query: Show the top 5 sectors by FDI inflow in USD for 2024-25.
        Query: What was the total FDI in INR crore for the Computer Software & Hardware sector in 2024-25?
        Query: List all sectors with an FDI inflow percentage greater than 10% in 2024-25.

        26. fdi_india_fy_country: This table provides annual Foreign Direct Investment (FDI) inflow data into India, broken down by country, with values in INR crore, USD million, and percentage share for each financial year.
        Instructions: Use this table to analyze FDI inflows into India by country and financial year, compare contributions, or track trends in FDI from different countries.
        Example Queries :
        Query: Show the top 5 countries by FDI inflow in USD for the financial year 2024-25.
        Query: List the FDI inflow percentage for each country in 2024-25.
        Query: What was the total FDI in INR crore received by India in 2024-25?
        Query: Show FDI inflow in USD million from the USA over all available years.

        27. energy_india_statewise_crude_oil_ngas_reserves: This table provides state-wise annual data on estimated reserves and distribution percentages of crude oil and natural gas in India.
        Instructions: Use this table to analyze or retrieve information about crude oil and natural gas reserves and their distribution across Indian states for specific years.
        Example Queries :
        Query: Show the crude oil and natural gas estimated reserves for Andhra Pradesh in 2024.
        Query: List all states with their crude oil distribution percent for the year 2023.
        Query: Get the natural gas estimated reserves and distribution percent for Arunachal Pradesh for all years.

        28. energy_india_statewise_coal_reserves: This table provides state-wise annual data on coal reserves in India, including proved, indicated, inferred, and total reserves, along with each state's percentage share of the national total.
        Instructions: Use this table to retrieve or analyze coal reserve statistics by state and year, such as total reserves, reserve categories, or distribution percentages. Filter by 'year' and 'state' as needed.
        Examples Queries :
        Query: Show the total coal reserves for each state in 2024.
        Query: Which state had the highest distribution percent of coal reserves in 2023?
        Query: List the proved, indicated, and inferred reserves for Andhra Pradesh over the years.

        29. crude_oil_mth_data: This table contains monthly crude oil data by company, including the month, year, oil company name, and the quantity in metric tonnes.
        Instructions: Use this table to analyze crude oil quantities by company, month, and year. Filter by 'month', 'year', or 'oil_company' to get specific data.
        Example Queries :
        Query: Show the total crude oil quantity for each company in 2022.
        Query: List the crude oil quantities for September 2022.
        Query: Find the total crude oil quantity for BPCL-KOCHI, KERALA across all years.

        30. ppac_mth_petroleum_consumption: Monthly petroleum product consumption data, including product type, quantity in metric tonnes, and reporting date.
        Instructions: Use this table to analyze monthly consumption trends of various petroleum products by month and year, or to retrieve quantities for specific products and time periods.
        Example Queries :
        Query: Show the total quantity of ATF consumed in 2023.
        Query: List the monthly consumption of all products for July 2023.
        Query: Get the latest updated date in the table.

        31. ores_minerals_exports_yearly: This table provides yearly export data for various ores and minerals, including quantities and values for the years 2015-16, 2016-17, and 2017-18, along with relevant measurement units and notes.
        Instructions: Use this table to analyze or retrieve export quantities and values for specific ores and minerals by year. You can filter by mineral name, year, or measurement unit, and review notes for data caveats.
        Example Queries :
        Query: Show the export value of 'Abrasive (Natural)' for 2017-18.
        Query: List all ores and minerals with their exported quantities in 2016-17.
        Query: Find the measurement unit used for 'Alabaster'.
        Query: Get the total export value for all minerals in 2015-16.
        Query: Show notes related to quantity for 'Alabaster' in 2015-16.

        32. mf_india_qtr_total: This table provides quarterly aggregated data on mutual fund schemes in India, including scheme names, number of schemes and folios, funds mobilized, redemptions, net inflows/outflows, assets under management, and other related metrics.
        Instructions: Use this table to analyze quarterly trends, compare mutual fund scheme categories, or summarize assets, inflows, and other key statistics for Indian mutual funds by scheme, quarter, or year.
        Example Queries :
        Query: Show the net inflow/outflow for each scheme in Q1 of 2025.
        Query: List the total number of folios for all schemes in June 2025.
        Query: Get the average net assets under management for each scheme type in 2025.
        Query: Which scheme had the highest funds mobilized in Q1 (Apr-Jun) 2025?

        33. insurance_india_mth_life_insurer → This table contains monthly performance data for life insurance companies in India, including premium values, growth percentages, and market share by insurer, category, and metric type.
        Instructions: Use this table to analyze and compare monthly life insurance metrics such as first year premium, growth, and market share across different insurers, categories, and time periods.
        Example Queries:
        Query: Show the total first year premium for all insurers in August 2025.
        Query: List the growth percentage for each insurer's Individual Non-Single Premium in August 2025.
        Query: Get the market share of Acko Life Insurance Limited for all categories in August 2025.

        34. trade_india_annual_country: This table shows the annual count of HS code tariff lines for different countries in trade with India.
        Instructions: Use this table to find the number of HS code tariff lines for a specific country and year, or to analyze trends in tariff lines over time for one or more countries.
        Example queries :
        Query: Show the total HS code tariff lines for USA in 2024.
        Query: List the annual HS code tariff lines for USA from 2022 to 2024.
        Query: Which countries had more than 20 HS code tariff lines in 2024?

        35. trade_india_mth_region_commodity: Monthly trade values between India and various regions and subregions, including actual and forecast data, broken down by month, year, and trade type.
        Instructions: Use this table to analyze India's trade values by region, subregion, month, year, and type (Actual or Forecast). Filter by region, subregion, month, year, or type as needed to get specific trade data.
        Example Queries :
        Query: Show the actual trade value for Europe in July 2024.
        Query: List forecasted trade values for all regions in July 2025.
        Query: Get the actual trade value for EU Countries in July 2024.

        36. port_dwell_time_month: This table contains monthly average dwell time data for imports and exports at various Indian port regions from 2021 to 2023, including details such as port name, month, year, category, and data source.
        Instructions: Use this table to analyze or retrieve port dwell time statistics by port region, month, year, and cargo category (Import/Export) for the years 2021 to 2023.
        Example Queries :
        Query: Show the average dwell time for imports at SOUTHERN REGION AVG. in January 2022.
        Query: List all export dwell times for EASTERN REGION AVG. in 2023.
        Query: What is the trend of import dwell times for SOUTHERN REGION AVG. from 2021 to 2023?

        37. trade_india_mth_country: This table contains monthly trade values between India and various countries, with details on country, year, month, and trade value.
        Instructions: Use this table to analyze India's trade values with specific countries by month and year. Filter by 'country', 'year', or 'month' as needed.
        Example Queries :
        Query: Show India's trade value with Afghanistan in July 2024.
        Query: List all countries and their trade values with India for July 2024.
        Query: Get the total trade value for all countries in 2024.

        38. trade_india_mth_region: Monthly trade values between India and different regions and subregions, including actual and forecast data.
        Instructions: Use this table to analyze India's trade values by region, subregion, month, year, and type (Actual or Forecast). Filter by these columns to get specific trade data.
        Example Queries :
        Query: Show the actual trade value for Europe in July 2024.
        Query: List forecasted trade values for all regions in July 2025.
        Query: Get trade values for EU Countries subregion in July 2024.

        39. cumulative_capacity_state_month: This table contains cumulative monthly renewable energy capacity data by Indian state, including various power sources such as wind, solar, hydro, and bioenergy.
        Instructions: Use this table to retrieve or analyze renewable energy capacity figures (in MW) for different Indian states, broken down by energy source and month/year.
        Example queries:
        Query: List all states with their solar power (gm) capacity for July 2025.
        Query: Show the total renewable energy capacity for Gujarat in July 2025.
        Query: Get the wind power and small hydro power for Rajasthan for July 2025.
        Query: Which state had the highest total renewable energy capacity in July 2025?
        Query: Show all columns for Tamil Nadu for July 2025.
        
        40. netc_india_mth_stats : This table contains monthly statistics for NETC (National Electronic Toll Collection) transactions in India, including transaction volumes and values, both total and average daily, for each month and year.
        Instructions: Use this table to analyze monthly trends in NETC transaction volumes and values, compare different months or years, or calculate averages and growth rates over time.
        Example queries:
        User Query: Show the total NETC transaction volume and value for each month in 2025.
        User Query: What was the average daily NETC transaction value in crores for August 2025?
        User Query: List the months in 2025 where the transaction volume exceeded 350 million.
        
        41. ins_nat_yr_fin_highlights : This table provides annual financial and operational highlights for insurance companies, including employee count, agents, offices, policies, FDI, and capital reserves by financial year.
        Instructions: Use this table to retrieve or analyze yearly statistics and financial data for insurance companies, such as number of employees, agents, offices, policies issued, FDI received, and capital reserves, filtered by company, category, or financial year.
        Example queries:
        User Query: Show the number of employees and policies for each insurance company in 2024-25.
        User Query: List all private sector insurers with their capital and free reserves for the latest year.
        User Query: Find companies with more than 10,000 agents or brokers in 2024-25.
        User Query: Get FDI received by each company in 2024-25.
        
        42. company_india_annual_type_mca : This table provides annual data on Indian companies, categorized by company type, including their paid-up share capital, number of companies, and their percentage share of the total companies for each financial year.
        Instructions: Use this table to analyze the distribution and capital structure of public and private limited companies in India across different financial years. Filter by 'financial_year' or 'company_type' to get specific insights.
        Example queries:
        User Query: Show the number of public limited companies for the year 2018-19.
        User Query: List the paid-up share capital for all company types in 2017-18.
        User Query: What percentage of total companies were private limited in 2019-20?
        User Query: Get the total number of companies for each year.
        
        43. energy_nat_ann_petroleum_products_consumption : This table contains annual and monthly consumption data of various petroleum products in India, including product type, quantity consumed (in thousand metric tonnes), and the date of data update.
        Instructions: Use this table to retrieve or analyze consumption quantities of specific petroleum products by month and year, or to compare trends across products and time periods.
        Example queries:
        User Query: Show the total consumption of ATF in 2023.
        User Query: List the monthly consumption of all petroleum products for August 2023.
        User Query: Get the latest updated date for the data.
        
        44. energy_wld_nat_mth_crude_petroleum_trade : Monthly trade statistics for crude petroleum and related products, including import/export quantities and values in INR and USD.
        Instructions: Use this table to analyze monthly import/export data for crude oil and petroleum products, including quantities, values, and product types by month and year.
        Example queries:
        User Query: Show total crude oil imports in metric tonnes for April 2025.
        User Query: List the USD value of LPG imports for each month in 2025.
        User Query: Get all petroleum product imports for April 2025 with their quantities and values.
        
        45. energy_nat_mth_crude_oil_prod : Monthly crude oil production data by company, including quantities in thousand metric tonnes.
        Instructions: Use this table to analyze or retrieve monthly crude oil production figures by company, year, and month.
        Example queries:
        User Query: Show total crude oil production for each company in 2023.
        User Query: List crude oil production for May 2023 by company.
        User Query: Get the monthly crude oil production trend for 'OIL' in 2023.
        
        46. energy_nat_mth_petroleum_prod : Monthly production data of various petroleum products, including product type, quantity produced (in thousand metric tonnes), and update date.
        Instructions: Use this table to retrieve, filter, or aggregate monthly petroleum production quantities by product, month, or year.
        Example queries:
        User Query: Show the total quantity of petroleum products produced in April 2023.
        User Query: List all products and their quantities for May 2023.
        User Query: Get the latest update date for the petroleum production data.
        
        47. mca_nat_mth_comp_closed : This table contains information about companies whose compliance for a particular month and year has been closed, including company details and relevant dates.
        Instructions: Use this table to find companies with closed compliance for specific months and years, or to track when compliance was released or updated.
        Example queries:
        User Query: Show all companies whose compliance was closed in September 2025.
        User Query: List the CIN and company names for companies with compliance released on 2025-10-01.
        User Query: Find all records updated after 2025-10-20.
        User Query: Get the details of companies from Karnataka whose compliance was closed in September 2025. 
        
        48. mca_nat_mth_companies_registered : This table contains details of companies registered in a given month, including their CIN, class, name, registration date, type, activity code, and relevant update dates.
        Instructions: Use this table to retrieve information about newly registered companies, filter by registration date, company class, type, or activity code, and analyze company registration trends.
        Example queries:
        User Query: Show all companies registered on 2025-04-01.
        User Query: List the names and CINs of all Private companies registered in April 2025.
        User Query: How many One Person Companies were registered in May 2025?
        User Query: Get all companies with activity code 41.00.
        
        49. none_of_these: Use this option if none of the above files contain the data required to answer the query.
        
        Consider the list above and respond only with one of the following file names:
        [irdai_nonlife_india_mth_insurer, fpi_india_yr_invtype, wages_sector_industry_index, eshram_state_dly_registrations, stock_india_mth_boaccounts, stock_india_mth_dps, fdi_state_qtr_view, revenue_maharashtra_fy_category, mf_monthly_schemes,
        statewise_petroleum_consumption, co2_emissions_by_fuel_yearly, quick_estimates_major_commodities_july_export, quick_estimates_major_commodities_july_import, statewise_cumulative_renewable_power, marketcap_nse_india_mth, ki_assam_mth_sctg,
        insurance_india_mth_sctg,toll_state_monthly_etc_transactions, ev_state_yr_catg, trade_india_mth_commodity_import, trade_india_mth_commodity_export, trade_india_mth_stateut_country_export, trade_india_mth_stateut_country_export,
        trade_india_mth_commoditygrp_country_export, fdi_india_fy_state, fdi_india_fy_sector, fdi_india_fy_country, energy_india_statewise_crude_oil_ngas_reserves, energy_india_statewise_coal_reserves,
        crude_oil_mth_data, ppac_mth_petroleum_consumption, ores_minerals_exports_yearly,mf_india_qtr_total, insurance_india_mth_life_insurer,trade_india_annual_country, trade_india_mth_region_commodity, port_dwell_time_month, trade_india_mth_country,
        trade_india_mth_region, cumulative_capacity_state_month, netc_india_mth_stats, ins_nat_yr_fin_highlights, company_india_annual_type_mca, energy_nat_ann_petroleum_products_consumption, energy_wld_nat_mth_crude_petroleum_trade, energy_nat_mth_crude_oil_prod,
        energy_nat_mth_petroleum_prod, mca_nat_mth_comp_closed, mca_nat_mth_companies_registered, none_of_these]
        Do not include any reasoning, explanation, or other text—only respond with the selected file name from the list above.

""")
    selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_agriculture_and_rural(query):
    system_instruction = dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        1. sa_agri_hhs_crop_sale_quantity_by_agency_major_disposal: This table presents state-wise and all-India data from the National Sample Survey Office on the distribution per 1000 of agricultural households reporting crop sales, and per 1000 quantity sold, by type of agency (APMC Market, Govt. Agency, FPOs, etc.) and crop (e.g., Wheat, Paddy, Potato, Maize). Information is collected for specific half-year periods (e.g., July-Dec 2018, Jan-Jun 2019) and distinguishes between different reporting rounds (visits).
        Sample queries:
        a. What proportion of paddy was sold through government agencies in Kerala during January to June 2019?
        b. For Madhya Pradesh, how many agricultural households reported selling urad through government agencies in July-December 2018?


        2. sa_agri_hhs_reporting_use_of_diff_farming_resources: This table provides state-wise and all-India data on the per 1000 agricultural households reporting use and procurement agency of various farming resources for different periods (e.g., July 2018-December 2018, January 2019-June 2019). It covers categories like animal feed, fertilizers, irrigation, manures, and veterinary services. Examples include Tamil Nadu (dry fodder from FPOs), Uttar Pradesh (chemical fertilizers from cooperatives), and Assam (animal feed). Frequency: semiannual (biannual) data.
        Sample queries:
        a. What percentage of agricultural households in West Bengal sourced irrigation resources from APMC Markets in January-June 2019?
        b. How many per 1000 agricultural households in Tamil Nadu obtained dry fodder through FPOs during January to June 2019?


        3. sa_agri_hhs_use_purchased_seed_by_quality: This table presents state-wise and Union Territory data from across India on the distribution per 1000 agricultural households reporting the use of purchased seed, categorized by quality (e.g., 'Good', 'Poor', 'Don’t Know') and agency of procurement (Input Dealers, Government Agency, Private Processor, etc.) for July 2018-December 2018 and January 2019-June 2019. Examples include Sikkim, Rajasthan (Input Dealers/Private Processor), and Punjab (Cooperatives). Data is semiannual, collected by NSSO, MoSPI.
        Sample queries:
        a. What is the per 1000 distribution of agricultural households in Mizoram that procured good quality seeds from private processors during July-December 2018?
        b. Which agency was the main source of good quality purchased seeds for Punjab agricultural households in January-June 2019?


        4. sa_avg_expenditure_and_receipts_on_farm_and_nonfarm_assets: This table presents state-wise and all-India data on average monthly expenditure and receipts for agricultural households, according to asset types (e.g., agricultural machinery, livestock, other assets), farm or non-farm business, and size class of land possessed. Data covers periods such as July 2018–June 2019 and January–June 2019. Examples include Andhra Pradesh, Kerala, Manipur, and all-India figures. Data is sourced from NSSO, MoSPI, and is reported at the state and national levels annually or semi-annually.
        Sample queries:
        a. What was the average monthly expenditure on agricultural machinery by households with 2.01-4.00 hectares in Karnataka during July 2018–June 2019?
        b. How does the average monthly receipt from the sale of livestock and poultry for farm business compare across Kerala and Mizoram during July 2018–June 2019?


        5. sa_avg_gross_cropped_area_value_quantity_crop_production: This table provides state-wise and all-India data on agricultural households, detailing average gross cropped area, value, and quantity of crop production for selected crops. Metrics include yield rates (e.g., Gujarat: 1,363 kg/ha), gross cropped area (e.g., Tamil Nadu: 0.581 ha), and value of production (e.g., Mizoram: Rs. 75,242). Data is biannual (July-December and January-June), differentiated by irrigation status, and is reported by NSSO/MoSPI.
        Sample queries:
        a. What was the average yield rate of harvested crops in Maharashtra for January 2019-June 2019 under irrigated conditions?
        b. How did the average gross cropped area per agricultural household compare between Tamil Nadu and Andhra Pradesh in July-December 2018?


        6. sa_avg_monthly_expenses_and_receipts_for_crop_production: This table details average monthly paid-out expenses and receipts for crop production across states of India and union territories, available at state-wise and group-of-states resolution. Data is classified by landholding size (e.g., '<0.01', '4.01-10.00' hectares), expenditure/receipt heads (like 'Irrigation', 'Labour-human', 'Electricity'), for periods between July 2018 and June 2019 (semi-annual or annual). Examples: Bihar's expenses on plant protection materials, Punjab's receipts, and Tripura's by-product receipts.
        Sample queries:
        a. What was the average monthly paid out expense on electricity for agricultural households in Maharashtra for July 2018–December 2018?
        b. Compare the average monthly receipts from harvested products in Manipur and Arunachal Pradesh for land holdings between 1.01 and 2.00 hectares.


        7. sa_avg_monthly_total_expenses_crop_production: This table provides state-wise and all-India monthly expenses (both paid out and imputed) for crop production per agricultural household, reported by the National Sample Survey Office. Data is available semi-annually (e.g., July 2018-December 2018) and disaggregated by land size class (e.g., 0.41-1.00 hectares). Expense categories include seeds, fertilizer, irrigation, interest, and lease rent. Examples: Assam’s labour-human expense (₹6157), West Bengal’s fertilizer/manure (₹858), Haryana’s plant protection materials (₹7323).
        Sample queries:
        a. What was the average monthly expense on fertilizer and manure for agricultural households in West Bengal during January-June 2019 for all land sizes?
        b. Can you provide total expenses for crop production by agricultural households with 1.01-2.00 hectares of land in Madhya Pradesh between July 2018 and December 2018?


        8. sa_avg_monthly_total_expenses_receipts_animal_farming_30_days: This table presents state-wise (e.g., Himachal Pradesh, Assam, Maharashtra) and group (e.g., Group of NE States) data on average monthly expenses and receipts (in INR) for farming of animals per agricultural household. Data is available by size class of land possessed (e.g., '<0.01', '10.00+'), cost categories (e.g., Animal feeds, Labour charges, Veterinary charges), and covers periods between July 2018 and June 2019, often semiannually. Examples: Punjab, Tripura, Kerala, Telangana.
        Sample queries:
        a. What was the average monthly expense on animal feeds for agricultural households in Karnataka with 1.01-2.00 hectares of land during July 2018–June 2019?
        b. How did labour charges for farming of animals differ for households with less than 0.01 hectares of land in Tripura and West Bengal in 2018-19?


        9. sa_dist_agri_hh_not_insuring_crop_by_reason_for_selected_crop: This table presents state and all-India level data on the distribution (per 1000) of agricultural households not insuring crops, segmented by reasons such as 'Not Aware about Availability of Facility', 'Delay in Claim Payment', 'Not Interested', and 'Complex Procedures'. Data covers multiple Indian states (e.g., Mizoram, Maharashtra, Arunachal Pradesh) and union territories, and is collected semi-annually (e.g., July-Dec 2018, Jan-Jun 2019) by the National Sample Survey Office, MoSPI.
        Sample queries:
        a. What were the main reasons for agricultural households in Karnataka not insuring their crops in January-June 2019?
        b. How does the rate of households not insuring crops due to 'Not Aware about Availability of Facility' compare between Telangana and West Bengal?


        10. sa_dist_agri_hhs_seed_use_by_agency_of_procurement: This table provides state-wise and all-India data on the distribution (per 1000 agricultural households) of agencies used for seed procurement for selected crops, sampled over two half-year periods (July-December 2018 and January-June 2019). Examples include procurement of paddy in Karnataka, coconut in Kerala, maize in Himachal Pradesh, and wheat in Chhattisgarh by agencies such as local markets, input dealers, government, or own farm. Data is collected by the National Sample Survey Office, MoSPI.
        Sample queries:
        a. What is the proportion of agricultural households in Gujarat that obtained Bajra seeds from the local market during January-June 2019?
        b. How many agricultural households at the all-India level used their own farm for Moong seed procurement in January-June 2019?


        11. sa_dist_hhs_leasing_out_land_and_avg_area_social_group: This table provides state-wise and all-India annual and half-yearly data (July 2018–June 2019, January–June 2019, etc.) on leasing-out of land by households, including metrics like number per 1000 households and average leased-out area (ha), disaggregated by social group (e.g., ST, OBC, SC, Others) and size class of ownership holding (e.g., 0.040-0.5 ha, All Sizes). Data sources include the National Sample Survey Office, MoSPI.
        Sample queries:
        a. What is the average area of land leased out per reporting household for OBCs in Maharashtra between January and June 2019?
        b. How many households per 1000 reported leasing out land in Assam (size class 0.040-0.5 ha) during July 2018–June 2019?


        12. sa_dist_of_agri_hhs_reporting_use_of_purchased_seed: This table presents state-wise and all-India data on the per 1000 distribution of agricultural households reporting the quality of purchased seeds for major crops. Data is available for individual states like Uttar Pradesh, Maharashtra, Arunachal Pradesh, and groups such as NE States, over two periods: July–December 2018 and January–June 2019 (biannual frequency). Reported seed quality categories include Good, Satisfactory, Poor, Don’t Know, N.r, and All. Source: National Sample Survey Office.
        Sample queries:
        a. What proportion of agricultural households in Odisha reported using good quality purchased seeds between January and June 2019?
        b. Compare the reported quality of purchased seeds among Maharashtra, Madhya Pradesh, and the national average for July–December 2018.


        13. sa_dist_of_hhs_by_hh_classification_for_diff_classes_of_land: This table presents all-India level data from the National Sample Survey Office (MoSPI) on per 1000 distribution of Scheduled Tribe (ST) agricultural households, classified by land possession size (e.g., '<0.01', '0.01-0.40' hectares) and principal source of income such as 'Casual Labour in Agriculture', 'Self-employment in Crop Production', and 'Regular Wage/Salary Earning in Non-agriculture'. Data refers to 'Visit 1' and is available for September 2023.
        Sample queries:
        a. What proportion of ST agricultural households with less than 0.01 hectares of land are engaged in non-agricultural casual labour?
        b. How many ST agricultural households possess 0.01-0.40 hectares and are primarily self-employed in crop production according to the 2023 survey?


        14. sa_distribution_hhs_leasing_in_land_avg_area_social_group: This table contains state-wise and all-India data on leasing-in and leasing-out of agricultural land for different household ownership size classes and social groups (e.g., SC, ST, OBC, Others, All) in India. Metrics include 'Average Leased-out Area per Reporting Household (ha)' and 'No. per 1000 of Households Reporting Leasing-out.' Data periods cover annual and half-yearly spans between July 2018 and June 2019. Examples: Tamil Nadu (SC, 0.040-0.5 ha), Chhattisgarh (Others, 2.0-3.0 ha), All India (SC, 0.5-1.0 ha).
        Sample queries:
        a. What is the average leased-out area per reporting household for OBC households in Andhra Pradesh in July 2018–June 2019?
        b. How many households per 1000 reported leasing-out land in Rajasthan for ST groups, and in which size class?


        15. sa_distribution_loan_outstanding_by_source_of_loan_taken: This table provides state-wise and all-India level data on the distribution of loan amounts outstanding among agricultural households, classified by the source of loan (e.g., Co-operative Society, Employer, NBFC Micro Finance, Commercial Bank) and landholding size (e.g., 0.41-1.00 ha, 10.00+ ha). The data represents the period July-December 2018 with observations for categories such as 'Average Amount of Outstanding Loan per Agricultural Household (Rs.)'. Source: National Sample Survey Office, MoSPI.
        Sample queries:
        a. What is the average amount of outstanding agricultural loans in Himachal Pradesh for landholding size 2.01 - 4.00 hectares?
        b. How many loans were outstanding from Co-operative Societies for agricultural households in Gujarat with 0.41 - 1.00 hectares of land during July to December 2018?


        16. sa_distribution_operational_holdings_by_possession_type: This table presents state-wise and all-India data on the per 1000 distribution of household operational holdings by type of possession (e.g., 'Entirely Owned', 'Entirely Leased in', 'Both Owned and Leased in', 'Entirely Otherwise Possessed') and size class (such as '0.040-0.5', '10.0-20.0', '>20.0') for various Indian states including Assam, Karnataka, and Bihar. Data is available semi-annually (July-December, January-June) for 2018-2019.
        Sample queries:
        a. What percentage of operational holdings in Assam were entirely owned during July 2018–June 2019 for the 4.0-5.0 hectare size class?
        b. Provide the all-India distribution for operational holdings that are 'Entirely Leased in' for the 5.0-7.5 hectare size class in January-June 2019.


        17. sa_est_num_of_hhs_for_each_size_class_of_land_possessed: This table presents state-wise estimated numbers of agricultural, non-agricultural, and all households in India by size class of land possessed (e.g., <0.01 ha, 4.01-10.00 ha, 10.00+ ha). Data is available for each visit (such as Visit 1, Visit 2) at the state/UT and group region level. Examples include Andhra Pradesh (all sizes, agricultural), Telangana (1.01-2.00 ha, all households), and Punjab (<0.01 ha, agricultural). Frequency is based on survey visits, typically biennial.
        Sample queries:
        a. What is the estimated number of agricultural households with land between 0.41 and 1.00 hectares in Odisha?
        b. How many non-agricultural households possessing all land sizes were there in Mizoram in Visit 1?


        18. sa_estimated_no_of_hhs_for_different_social_groups: This table provides estimated numbers of households in India by state, household type (agricultural, non-agricultural, all), and social group (SC, ST, OBC, Others, All). Data is available at the national, state, and group-of-states level, with visits (rounds) such as 'Visit 1' and 'Visit 2'. Examples include OBC households in Bihar, SC households in Tripura, and all households in Sikkim. Data frequency appears to be cross-sectional (by visit/round).
        Sample queries:
        a. What is the estimated number of non-agricultural SC households in Tripura according to Visit 1?
        b. How many agricultural OBC households were recorded in Arunachal Pradesh during Visit 1?


        19. sa_no_of_hhs_owning_of_livestock_of_different_types: This table presents state-wise, all-India, and group-of-state/UTs data on the number of households owning livestock per 1000 households, categorized by social groups (e.g., SC, ST, OBC, Others), gender (Male, Female, Person), and size classes of household operational holdings (e.g., 0.040-0.5, 2.0-3.0, >20.0 hectares). Data examples include Telangana (2.0-3.0 ha, SC), Uttar Pradesh (7.5-10.0 ha, Female, Young Stock), and All India (ST, 0.5-1.0 ha). The time frequency is semi-annual (July–December 2018).
        Sample queries:
        a. How many OBC households in Andhra Pradesh owning operational holdings of 1.0-2.0 hectares reported livestock ownership?
        b. What is the number of households per 1000 owning young livestock among SCs in Haryana with 1.0-2.0 hectare holdings in July–December 2018?


        20. sa_no_per_1000_distri_of_agri_hhs_reporting_sale_of_crops: This table presents state-wise and all-India data from the National Sample Survey Office on agricultural households’ awareness and experience with Minimum Support Price (MSP) schemes, covering aspects like awareness, output sold under MSP, challenges (e.g., insurance facility not available, complex procedures), and average sale rates. Data is available for states such as Rajasthan, Bihar, Haryana, and Kerala, and is reported semi-annually (e.g., Jan-Jun 2019, Jul-Dec 2018) for selected visits.
        Sample queries:
        a. What percentage of agricultural output was sold under MSP in Tamil Nadu during July 2018–December 2018?
        b. How many agricultural households in Bihar were aware of insurance facility unavailability under MSP between January and June 2019?


        21. sa_no_per_hh_operational_holding_by_size_hh_oper_holding: This table provides state-wise and all-India level data on household operational holdings in India, covering various metrics such as average area owned, area operated, number of crops harvested, and joint holdings. Data is available for agricultural and non-agricultural households, across social groups (SC, ST, OBC, Others) and different holding size classes. Time frequency is semi-annual, for periods like July-December 2018 and January-June 2019. Examples include Kerala, Bihar, and All India.
        Sample queries:
        a. What was the average area operated per holding for agricultural households in Madhya Pradesh during January-June 2019?
        b. How many joint holdings per 1000 were reported for non-agricultural households in Tripura in January-June 2019?


        22. sa_per_1000_agri_hh_insured_experienced_crop_loss: This table provides state-wise and all-India statistics on the number per 1000 agricultural households that were additionally insured and experienced crop loss for selected crops (e.g., Maize, Bajra, Arhar, Potato) during July 2018–June 2019 (reported in Visit 1 and Visit 2). It details claim status categories, such as 'Received claim fully', 'Not received claim due to cause outside coverage', and includes data for various states like Maharashtra, West Bengal, Sikkim, and data at the All India level. Data frequency: semi-annual.
        Sample queries:
        a. What proportion of agricultural households in Maharashtra that insured additionally and experienced crop loss received insurance claim fully for cotton in July–December 2018?
        b. How many per 1000 agricultural households at the all-India level received insurance claim partly for potato crop loss during January–June 2019?


        23. sa_per_1000_crop_producing_hh_crop_disposal_agency_sale_satisf: This table presents state-wise data on the number per 1000 crop-producing agricultural households reporting crop disposals to various agencies (e.g., APMC Market, Cooperatives, FPOs, Private Processors) by their level of satisfaction with sale outcomes (e.g., Satisfactory, Not Satisfactory). It covers periods like January–June 2019 and July–December 2018, with categories for dissatisfaction reasons such as 'Lower than Market Price' and 'Delayed Payments.' Data are reported semi-annually for states including Kerala, Jharkhand, and Punjab.
        Sample queries:
        a. Which states reported the highest proportion of agricultural households dissatisfied due to lower than market price during January to June 2019?
        b. What is the distribution of satisfaction levels among crop-producing households selling to private processors in Karnataka and Telangana?


        24. sa_perc_dist_of_land_for_hhs_belonging_operational_holding: This table presents the percentage distribution of land area among rural households by operational holding size and type of land possession (owned, leased, otherwise possessed) across India, states, and groups of states/UTs. Data is state-wise and all-India, based on the National Sample Survey Office (NSSO) for July 2018–June 2019. Examples include Andhra Pradesh (owned and possessed: 63.2%), Odisha (leased out: 8.4%), and Sikkim (otherwise possessed: 0.0%).
        Sample queries:
        a. What percentage of land in Punjab for households holding 1.0-2.0 hectares is leased in?
        b. Which state had the highest proportion of operational holdings owned and possessed in the 3.0-4.0 hectare class?


        25. sa_percent_distribution_of_leased_out_land_by_terms_of_lease: This table presents state-wise and group-level data for India on the percentage distribution and area of leased-out land, classified by terms of lease (e.g., Fixed Money, Share of Produce, Fixed Produce), household holding size (such as 1.0-2.0 ha, 7.5-10.0 ha), and lease terms. Data are available for various states (Bihar, Tamil Nadu, Rajasthan), certain periods (July 2018-December 2018, January 2019-June 2019), and include national and group aggregates. The data frequency is semi-annual.
        Sample queries:
        a. What percentage of leased-out land in Bihar was under fixed produce terms for all household sizes in July-December 2018?
        b. How many households per 1000 reported leasing out land of size 1.0-2.0 ha in Rajasthan during January-June 2019?
        
        26. annual_mean_temperature : This table contains annual and seasonal mean temperature data by year, including temperature values for different periods within each year.
        Instructions: Use this table to retrieve mean temperature values for specific years, periods (such as annual or seasonal), or to analyze temperature trends over time.
        Example queries:
        User Query: What was the annual mean temperature in 2024?
        User Query: Show the mean temperature for each period in 2024.
        User Query: List all available years in the dataset.
        User Query: Get the mean temperature for Mar-May across all years.
        
        27. fish_production_yearly : This table contains yearly fish production data for Indian states and union territories, including the year, indicator, unit, state, and production value.
        Instructions: Use this table to retrieve or analyze fish production figures by year, state, or other attributes such as unit or indicator.
        Example queries:
        User Query: Show total fish production for Andhra Pradesh in 2023-24.
        User Query: List all states with their fish production values for the year 2023-24.
        User Query: What was the fish production in Arunachal Pradesh in 2023-24?
        
        28. watersheds_in_india : This table contains data on watersheds in India, including year, indicator, sub-indicator, river length, sub-basin, and corresponding values such as area in square kilometers.
        Instructions: Use this table to retrieve information about watershed areas, river lengths, sub-basins, and related indicators for different years in India.
        Example queries:
        User Query: Show the area in square kilometers for each sub-basin of the Indus river in 2019.
        User Query: List all available sub-indicators for the year 2019.
        User Query: Get the total area covered by all sub-basins of the Indus river in 2019.
        
        29. river_basin_catchment : This table contains data on the catchment areas of major river basins, including details such as year, indicator, unit, sub-indicator, river name, and the corresponding value.
        Instructions: Use this table to retrieve information about the catchment area sizes of various river basins by year, river name, or other related attributes.
        Example queries:
        User Query: Show the catchment area for all rivers in 2024.
        User Query: List the catchment area of the Indus (Eastern) river.
        User Query: What is the total catchment area covered by all rivers in the latest year?
        
        30. coastline_population_and_length : This table provides data on the length of coastline and the coastal population for different states of India by year, including specific sub-indicators for population and coastline length.
        Instructions: Use this table to retrieve information about the coastal population or coastline length for Indian states for specific years or to compare these metrics across states.
        Example queries:
        User Query: What was the coastal population of Andhra Pradesh in 2025?
        User Query: List the coastline length for all states in 2025.
        User Query: Show the coastal population and coastline length for Andaman and Nicobar Islands in 2025.
        
        31. rainfall_annualy_and_monthly : This table contains annual and monthly rainfall data, including the year, month, rainfall value, and measurement unit.
        Instructions: Use this table to retrieve rainfall statistics by year and month, filter by specific periods, or aggregate rainfall values.
        Example queries:
        User Query: Show the total rainfall for each month in 2024.
        User Query: Get the rainfall value for March 2024.
        User Query: List all available years in the rainfall data.
        
        32. faunal_diversity : This table provides data on the diversity of marine faunal species in India and the world, categorized by indicator, unit, sub-indicator, phylum, and numeric value.
        Instructions: Use this table to retrieve information about the number and types of marine faunal species, grouped by various taxonomic phyla and indicators. Filter by indicator, sub-indicator, or phylum as needed.
        Example queries:
        User Query: Show the total number of marine faunal species of world for each phylum.
        User Query: List all available sub-indicators in the faunal diversity data.
        User Query: Get the value for PROTOZOA under the indicator 'Faunal diversity in India and the world'.  annual_mean_temperature, fish_production_yearly, watersheds_in_india, river_basin_catchment, coastline_population_and_length, rainfall_annualy_and_monthly, faunal_diversity
        
        
        23. none_of_these: This should be used when the query is unrelated to agricultural household crop sales, crop-selling agencies, or seasonal crop marketing patterns. For example, queries about GDP, inflation, government policies, crop production volume, weather, or support prices (MSP) fall under this category.

        Consider the list above and respond only with one of the following file names:
        [sa_agri_hhs_crop_sale_quantity_by_agency_major_disposal, sa_agri_hhs_reporting_use_of_diff_farming_resources, sa_agri_hhs_use_purchased_seed_by_quality, sa_avg_expenditure_and_receipts_on_farm_and_nonfarm_assets, sa_avg_gross_cropped_area_value_quantity_crop_production, sa_avg_monthly_expenses_and_receipts_for_crop_production, sa_avg_monthly_total_expenses_crop_production, sa_avg_monthly_total_expenses_receipts_animal_farming_30_days,
        sa_dist_agri_hh_not_insuring_crop_by_reason_for_selected_crop, sa_dist_agri_hhs_seed_use_by_agency_of_procurement, sa_dist_hhs_leasing_out_land_and_avg_area_social_group, sa_dist_of_agri_hhs_reporting_use_of_purchased_seed, sa_dist_of_hhs_by_hh_classification_for_diff_classes_of_land, sa_distribution_hhs_leasing_in_land_avg_area_social_group, sa_distribution_loan_outstanding_by_source_of_loan_taken, sa_distribution_operational_holdings_by_possession_type,
        sa_est_num_of_hhs_for_each_size_class_of_land_possessed, sa_estimated_no_of_hhs_for_different_social_groups, sa_no_of_hhs_owning_of_livestock_of_different_types, sa_no_per_1000_distri_of_agri_hhs_reporting_sale_of_crops, sa_no_per_hh_operational_holding_by_size_hh_oper_holding, sa_per_1000_agri_hh_insured_experienced_crop_loss, sa_per_1000_crop_producing_hh_crop_disposal_agency_sale_satisf, sa_perc_dist_of_land_for_hhs_belonging_operational_holding,
        sa_percent_distribution_of_leased_out_land_by_terms_of_lease,annual_mean_temperature, fish_production_yearly, watersheds_in_india, river_basin_catchment, coastline_population_and_length, rainfall_annualy_and_monthly, faunal_diversity, none_of_these]
        Do not include any reasoning, explanation, or other text—only respond with the selected file name from the list above.
    """)
    selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_enterprise_surveys(query):
    system_instruction=dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        1. asuse_est_num_establishments_pursuing_mixed_activity: This table presents annual, all-India estimates of establishments pursuing mixed activities, disaggregated by sector (urban, rural, combined), establishment type (Own Account, Hired Worker, All), and economic activity (e.g., Tobacco Products, Professional Services, Real Estate, Food Manufacturing). Data come from the National Sample Survey Office, MoSPI. Examples: 452,291 urban OAEs in tobacco manufacture (2022-23), 10,336,002 rural retail trade establishments (2023-24), and 24,643,235 all-India service providers (2022-23).
        Sample queries:
        a. How many rural own account establishments were engaged in real estate activities in 2022-23?
        b. What is the estimated number of establishments involved in 'Other Retail Trade' at all-India level for 2023-24?

        2. asuse_per1000_estb_by_hours_worked_per_day: This table reports annual, All-India level data on the per-1000 distribution of establishments by hours worked per day, categorized by rural/urban/combined sectors and establishment types such as Own Account Establishments and Hired Worker Establishments. Industry categories include 'Manufacture of Beverages', 'Trading Activities', 'Financial Services', and 'Food and Accommodation Service Activities', with working hours grouped as '<4', '4-7', '8-11', '>11', and 'All'. Example: 705 establishments (urban, professional activities) work '8-11' hours.
        Sample queries:
        a. What percentage of rural establishments in the Manufacture of Beverages category worked more than 11 hours a day in 2023-24?
        b. How does the distribution of working hours differ between urban and rural establishments in the Financial Service Activities Except Insurance and Pension Funding sector in 2022-23?

        3. asuse_per1000_estb_by_months_operated_last_365days: This table presents annual all-India data on the distribution of establishments by months of operation, disaggregated by sector (e.g., Manufacture of Electrical Equipment, Real Estate Activities, Water Transport), type (Own Account, Hired Worker), and rural/urban/combined geographies. Key metrics include per-1000 distribution or average months operated for categories such as '<= 3 Months', '7 to 9 Months', and '> 9 Months'. Data is sourced from the National Sample Survey Office (MoSPI).
        Sample queries:
        a. What was the average number of months operated by rural establishments engaged in the manufacture of leather and related products in 2023-24?
        b. How many per 1000 urban hired worker establishments in wholesale on a fee or contract basis operated for less than or equal to 3 months in 2021-22?

        4. asuse_per1000_estb_registered_under_acts_authorities: This table provides annual, all-India data on the number per 1000 of establishments registered under various Acts or authorities (e.g., Societies Reg. Act, CGST Act, EPFO/ESIC, RTO). The data is category-wise (urban, rural, combined), sector-wise (e.g., Manufacture of Textiles, Information and Communication, Trading Activities), and by type of establishment (All/Own Account/Hired Worker). For example, in 2021-22, 277 per 1000 urban hired worker establishments in wood manufacturing were registered under Shops & Establishments Act.
        Sample queries:
        a. What was the number per 1000 of rural establishments registered under the CGST Act for 'Wholesale and Retail Trade of Motor Vehicles and Motor Cycles' in 2021-22?
        b. How did registration of hired worker establishments in 'Non-captive Electricity Generation and Transmission' change between 2022-23 and 2023-24 under 'Others' in urban areas?

        5. asuse_per1000_estb_using_computer_internet_last365_days: This table presents annual, all-India data from the National Sample Survey Office on the number per 1000 establishments using computers and internet during the last 365 days. The data is available by sector (Trade, Manufacturing, Other Services), location (Rural, Urban, Combined), and type (Own Account Establishments, Hired Worker Establishments, All). For example, in 2023-24, 129 per 1000 urban service establishments used computers, while 640 per 1000 hired worker trade establishments used internet.
        Sample queries:
        a. What was the number per 1000 manufacturing establishments using the internet in urban areas in 2023-24?
        b. How many per 1000 own account trade establishments in rural areas used computers in 2022-23?

        6. asuse_est_annual_gva_per_establishment: This table provides estimated annual Gross Value Added (GVA (Gross Value Added)) per establishment, categorized by year, state/UT, sector, activity category, and establishment type, sourced from the National Sample Survey Office.
        Instructions: Use this table to analyze or compare the estimated annual GVA (Gross Value Added) per establishment across different years, states/UTs, sectors (rural/urban), activity categories, and establishment types (such as Hired Worker Establishments or Own Account Establishments). Filter by relevant columns to get specific insights.
        Example queries:
        Query: What was the estimated annual GVA (Gross Value Added) per establishment for Cotton Ginning, Cleaning and Bailing in rural All India for 2023-24?
        Query: Show the GVA (Gross Value Added) per establishment for all establishment types in 2023-24 for rural sector.
        Query: List the years and GVA (Gross Value Added) per establishment for Hired Worker Establishments in Cotton Ginning, Cleaning and Bailing activity.

        7. asuse_per1000_of_estb_using_internet_by_type_of_its_use: This table provides annual, all-India statistics on the number per 1,000 of establishments using the Internet, broken down by sector (e.g., Manufacturing, Trade, Other Services), area (Rural, Urban, Combined), and specific types of Internet use (such as Internet Banking, Delivering Products Online, Telephoning Over VoIP, Customer Services). Example entries include: Manufacturing establishments in rural India using the Internet for government information in 2022-23 (46), or Urban Trade sector using it for staff training in 2021-22 (323).
        Sample queries:
        a. How many establishments per 1,000 in the Trade sector used Internet banking in urban India in 2021-22?
        b. What was the number per 1,000 of rural manufacturing establishments using the Internet for accessing financial services in 2023-24?

        8. asuse_per1000_proppartn_estb_by_edu_owner_mjr_partner: This dataset provides the annual (financial year) per 1000 distribution of proprietary and partnership establishments in India, categorized by the level of general education of the owner or major partner. Data is available at the all-India level, disaggregated by rural, urban, and combined sectors. Education categories include Not Literate, Literate Below Primary, Literate Graduate and Above, among others. Example: 199 urban establishments per 1000 had graduate owners in 2022-23.
        Sample queries:
        a. What percentage of rural proprietary establishments in 2022-23 were owned by people with primary to below secondary education?
        b. How did the distribution of urban establishments owned by graduates change between 2021-22 and 2023-24?

        9. asuse_per1000_proppartn_estb_by_other_econ_activities: This table provides annual, all-India level data on the distribution per 1,000 Proprietary and Partnership establishments by various economic activities (e.g., Manufacturing Activities, Food and Accommodation Service Activities, Real Estate, Education). Data is disaggregated by rural/urban/combined sectors, establishment size (e.g., Hired Worker Establishments, Own Account Establishments), and the number of other economic activities present. Data spans years like 2021-22 to 2023-24. Source: National Sample Survey Office, MoSPI.
        Sample queries:
        a. What was the per 1000 distribution of proprietary and partnership establishments for Manufacture of Pharmaceuticals in urban India in 2022-23?
        b. Show the annual trend from 2021-22 to 2023-24 for All Establishments in Food and Accommodation Service Activities at the all-India level.

        10. asuse_per1000_proppartn_estb_by_socialgroup_owner: This table presents all-India level, annual data on the per 1000 distribution of proprietary and partnership establishments by the social group of owner or major partner. Data is provided across urban, rural, and combined sectors, for various establishment types (Own Account, Hired Worker, All), and industries such as Manufacture of Rubber and Plastics Products, Accommodation, and Trading Activities. Social group categories include Scheduled Tribe, Scheduled Caste, OBC, Others, and Not Known.
        Sample queries:
        a. What is the distribution of hired worker establishments owned by Scheduled Tribe groups in the financial service sector for 2022-23?
        b. How do the per 1000 establishment distributions for the manufacture of paper and paper products differ between urban and rural areas for OAE in 2023-24?

        11. asuse_per_1000_distri_of_establishments_by_nature_of_operation: This table presents annual all-India estimates of the per 1000 distribution of establishments by nature of operation, disaggregated by sector (e.g., manufacture of food products, education, retail trade), type of establishment (Own Account, Hired Worker, All), and operational status (Perennial, Seasonal, Casual), for rural, urban, and combined regions. Data examples include 'Manufacture of Furniture' (urban, all), 'Other Wholesale Trade' (rural, HWE, seasonal), and 'Human Health and Social Work Activities'.
        Sample queries:
        a. What percentage of urban establishments in India engaged in manufacture of furniture are perennial according to the latest available year?
        b. How does the distribution of hired worker establishments in the education sector vary between rural and urban areas for 2022-23?

        12. asuse_per_1000_distri_of_establishments_by_type_of_location: This table presents annual all-India data on the per 1000 distribution of establishments by type of location, categorized by rural, urban, and combined regions. Industries covered range from 'Other Manufacturing', 'Land Transport', 'Education', 'Financial Service Activities' to 'Manufacture of Chemicals'. Data is further detailed for own account, hired worker, and all establishments across location types such as household premises, permanent and temporary structures, and mobile markets.
        Sample queries:
        a. What is the distribution of 'Land Transport' establishments located outside household premises with permanent structure in rural areas for 2021-22?
        b. How many 'Manufacture of Chemicals and Chemical Products' own account establishments operated within household premises in urban India in 2023-24 per 1000 units?

        13. asuse_per_1000_distri_of_establishments_by_type_of_ownership: This table presents all-India annual data on the per-1000 distribution of establishments by ownership type, for different activity categories (e.g., Manufacture of Textiles, Trading Activities, Accommodation) and sectors (urban, rural, combined). Ownership types include Proprietary-Male, SHG, Partnership, Co-operatives, among others. For example, in 2023-24, 908 per 1000 urban 'Other Financial Activities' establishments were 'Proprietary-Male', and 1000 per 1000 urban 'Accommodation' HWE were 'All'.
        Sample queries:
        a. What percentage of urban establishments under 'Manufacture of Tobacco Products' are owned by women in 2023-24?
        b. How does the distribution of ownership type in 'Accommodation' activities differ between rural and urban India for 2022-23?

        14. asuse_per_1000_of_establishments_which_are_npis_and_non_npis: This table presents annual state-wise data on the number per 1000 of establishments classified as NPIs (Non-Profit Institutions) and non-NPIs, disaggregated by urban, rural, and combined sectors across India. Categories include 'Trade', 'Manufacturing', and 'Other Services', with receipt sources such as 'Donation/Grants' and 'Other Sources'. Examples include 947 non-NPIs per 1000 establishments in urban Nagaland (2023-24) and all-India combined data for manufacturing in 2022-23.
        Sample queries:
        a. What percentage of trade establishments were NPIs with major receipts from donations in Rajasthan (urban) in 2021-22?
        b. Show state-wise data for non-NPI establishments in other services for the year 2022-23.

        15. asuse_statewise_est_num_of_estb_pursuing_mixed_activity: This table provides annual state/UT-wise data on the estimated number of establishments pursuing mixed activities in India, disaggregated by sector (e.g., Manufacturing, Trade, Other Services), area (Urban, Rural, Combined), and establishment type (Own Account Establishments, Hired Worker Establishments, All). Data examples include 763,860 rural manufacturing establishments in Maharashtra (2022-23), 9,483 urban HWE in Chandigarh (2022-23), and 2,197,497 combined OAE in Odisha (2021-22).
        Sample queries:
        a. How many urban hired worker establishments pursuing mixed activities were there in Uttarakhand in 2023-24?
        b. Provide the number of manufacturing establishments in Tamil Nadu (all establishment types) for 2022-23, broken down by area.

        16. asuse_statewise_est_num_of_estb_serving_as_franchisee_outlet: This table contains annual state/UT-wise estimates of the number of establishments serving as franchisee outlets across India from 2021-22 to 2023-24, with data provided at both state (e.g., Bihar: 1129; Odisha: 5826; Telangana: 7308) and all-India levels (e.g., 132471 in 2023-24). Data covers all major states and union territories, including minor entries (e.g., Lakshadweep: 0), and is sourced from the National Sample Survey Office, MoSPI.
        Sample queries:
        a. How many franchisee outlets were estimated in Tamil Nadu in each of the last three years?
        b. Which Indian state had the highest number of franchisee establishments in 2022-23 according to the NSSO?

        17. asuse_statewise_estimated_annual_gva_per_establishment_rupees: This table provides state and sector-wise estimated annual Gross Value Added (GVA (Gross Value Added)) per establishment (in Rupees) for different establishment types and broad activity categories, as reported by the National Sample Survey Office.
        Instructions: Use this table to analyze or retrieve estimated annual GVA (Gross Value Added) per establishment by state/UT, year, sector (rural/urban), establishment type (HWE/OAE/All), and broad activity category (e.g., Manufacturing). Filter by relevant columns to get specific GVA (Gross Value Added) values.
        Example queries:
        Query: Show the estimated annual GVA (Gross Value Added) per establishment for all establishment types in Andhra Pradesh (rural, manufacturing) for 2023-24.
        Query: List the GVA (Gross Value Added) per establishment for each state in the manufacturing sector for 2023-24 (all establishment types, rural only).
        Query: Get the GVA (Gross Value Added) per establishment for Own Account Establishments in Andhra Pradesh for 2023-24, manufacturing sector, rural area.

        18. asuse_statewise_per1000_distri_of_estb_by_nature_of_operation: This table provides state/UT-wise, rural/urban/combined, and all-India level annual data on the per 1000 distribution of establishments by nature of operation (perennial, seasonal, casual). Categories covered include Manufacturing, Trade, Other Services, and All, with breakdowns for Own Account Establishments (OAE), Hired Worker Establishments (HWE), and all establishments. Example entries: Uttar Pradesh, Gujarat, Sikkim, Puduchery, Maharashtra (2021–2024, all modes and categories).
        Sample queries:
        a. What percentage of manufacturing establishments in rural Uttar Pradesh were perennial in 2021-22?
        b. Show the state-wise distribution of seasonal own account establishments in the trade sector for 2022-23.

        19. asuse_statewise_per1000_distri_of_estb_by_type_of_location: This table presents annual, state/UT-wise data on the per-1000 distribution of establishments in India, categorized by sector (Manufacturing, Trade, Other Services), establishment type (Own Account Establishments, Hired Worker Establishments, All), and location type (e.g., within household premises, street vendors, kiosks). Coverage includes states like Tamil Nadu, Gujarat, West Bengal, and all-India aggregates, with data available separately for rural, urban, and combined geographies from 2021-22 to 2023-24.
        Sample queries:
        a. What is the distribution of manufacturing establishments by location type in rural West Bengal for 2023-24?
        b. How does the per-1000 share of street vendor establishments in urban Gujarat compare between 2021-22 and 2023-24?

        20. asuse_statewise_per1000_distri_of_estb_by_type_of_ownership: This table provides annual, state/UT-wise and all-India data on the per-1000 distribution of establishments by type of ownership, segmented by rural, urban, and combined areas. Categories include Manufacturing, Trade, and Other Services, with detailed breakdowns such as Proprietary-Male, Partnerships, Societies, and SHGs. Examples include Assam (Rural, Trade), Punjab (Urban, Other Services), and all-India rural data for Trade. Source: National Sample Survey Office, MoSPI.
        Sample queries:
        a. What is the share of proprietary-female owned establishments in the 'Other Services' sector in urban Punjab for 2023-24?
        b. Show per 1000 distribution of SHG-owned establishments for all establishment types in rural Kerala for 2022-23.

        21. asuse_statewise_per1000_estb_by_hours_worked_per_day: This table provides annual, state/UT-wise data on the per-1000 distribution of establishments by the number of hours normally worked per day. It covers various states (e.g., Goa, Maharashtra, Gujarat), sectors (Trade, Manufacturing, Other Services), establishment types (Own Account, Hired Worker, All Establishments), and urban/rural status. Examples include 813 Own Account Trade establishments in Goa and 815 Hired Worker Trade establishments in Assam (2023-24).
        Sample queries:
        a. What is the distribution of manufacturing establishments by hours worked in a day for Telangana (urban) in 2023-24?
        b. Compare the proportion of Own Account versus Hired Worker establishments working 8-11 hours in Gujarat during 2022-23.

        22. asuse_statewise_per1000_estb_by_month_num_operated_last365_day: This table presents annual data on the distribution per 1000 of establishments by number of months operated in the last 365 days, covering different states (e.g., Odisha, Karnataka, Delhi), sectors (Trade, Manufacturing, Other Services), and establishment types (Own Account, Hired Worker, All). Data is available at all-India, state, and urban/rural/combined levels with examples like Odisha (Rural, Trade, <=3 Months) and Delhi (Combined, Manufacturing, <=3 Months).
        Sample queries:
        a. How many own account establishments in Haryana traded for <=3 months during 2023-24?
        b. Which state had the highest per 1000 distribution of trade establishments operating more than 9 months in rural areas in 2022-23?

        23. asuse_statewise_per1000_estb_maintain_post_bank_saving_acc: This table presents annual, state/UT-wise and location-wise (urban/rural/combined) data on the number per 1000 establishments maintaining bank or post office savings accounts in India, split by sectors such as Trade, Manufacturing, and Other Services. Data is categorized for Own Account and Hired Worker Establishments, and by account holder type. Examples include Maharashtra-rural (317, HWE), Assam-urban (914, OAE), and Chandigarh-rural (1000, HWE), from 2021-22 to 2023-24.
        Sample queries:
        a. What was the number per 1000 of urban establishments in Karnataka maintaining any bank account in 2021-22?
        b. How does the proportion of rural own account establishments with Post Office Savings Bank accounts in Punjab compare to Haryana in 2023-24?

        24. asuse_statewise_per1000_estb_registered_diff_acts_authorities: This table provides annual, state/UT-wise and all-India statistics on the number per 1000 of establishments registered under various Acts and authorities (e.g., Shops & Establishment Act, EPFO/ESIC, Co-operative Societies Act) by sector (Trade, Manufacturing, Other Services) and type (Own Account, Hired Worker, All). Examples include 0 per 1000 for 'EPFO/ESIC' in Goa Rural (2021-22) and 148 per 1000 for 'Others' in All India Rural (2022-23).
        Sample queries:
        a. What is the number per 1000 of establishments registered under the Shops & Establishment Act in urban Odisha for 2021-22?
        b. Provide state-wise data for 2023-24 on establishments registered under the Co-operative Societies Act, 1912 in the 'Other Services' sector.

        25. asuse_statewise_per1000_estb_use_computer_internet_last365_day: This table presents annual, state/UT-wise data on the number per 1000 establishments using computers and internet in India, across years like 2021-22 and 2023-24. Data is disaggregated by sector (e.g., Trade, Manufacturing, Other Services), type of establishment (Own Account, Hired Worker, All), and area (Urban, Rural, Combined). Examples include Kerala (rural, Hired Worker, computer: 278), Chandigarh (Combined, Hired Worker, internet: 958), and Uttar Pradesh (urban, all, internet: 158).
        Sample queries:
        a. What is the number per 1000 of trade establishments using computers in urban Sikkim for 2023-24?
        b. How many own account establishments in rural Madhya Pradesh used the internet per 1000 in 2022-23?

        26. asuse_statewise_per1000_proppart_estb_by_social_grp_mjr_prtner: This table provides annual state/UT-wise and national-level data on the per 1000 distribution of proprietary and partnership establishments in India by the social group of the owner-major partner. It covers rural, urban, and combined geographies, establishment types (Own Account, Hired Worker, All), sectors like Manufacturing, Trade, and Other Services, and social groups including Scheduled Castes, Scheduled Tribes, Other Backward Classes, and Others. Example entries: Rajasthan, 2021-22, Manufacturing, Scheduled Caste; All India, 2023-24, Manufacturing, OBC.
        Sample queries:
        a. What is the proportion of Scheduled Tribe-owned manufacturing establishments in Meghalaya (rural) in 2023-24?
        b. How did the distribution of Other Backward Classes in trade sector establishments change in Andhra Pradesh from 2021-22 to 2023-24?

        27. annual_survey_of_industries: This table contains Annual Survey of Industries (ASI) data with comprehensive industrial performance metrics for India.
        It includes:
        **Financial Metrics**: Addition in Stock (Materials, Finished Goods, Semi-Finished Goods), Depreciation, Gross/Net Capital Formation, Gross/Net Value Added, Interest Paid/Received, Invested Capital, Net Income, Net Profit, Outstanding Loan, Physical Working Capital, Rent Paid/Received, Total Inputs/Output, Working Capital
        **Employment & Labor**: Bonus to All Staff, Employers' Contribution, No. of Directly Employed Workers (Male/Female), No. of Employees Other Than Workers, No. of Workers Employed Through Contractors, Total Mandays Employed, Total Number of Persons Engaged, Wages and Salaries (with various breakdowns)
        **Production & Operations**: Factories in Operation, Number of Factories, Fuels Consumed (Coal, Electricity, Petroleum, Other), Materials Consumed, Value of Product and By-Product
        **Infrastructure**: Fixed Capital, Gross Value of Plant & Machinery, Quantity of Coal/Electricity Consumed
        Use this table for queries about: industrial establishments, factory data, manufacturing statistics, capital investment, employment in industries, wages, production output, industrial finance, depreciation, bonus payments, factory operations, industrial sectors, manufacturing performance.

        **Query Classification Guidelines:**
        - If query mentions "industries", "factories", "manufacturing", "industrial sectors", "production", "depreciation", "bonus", "wages in industries", "capital formation", "industrial employment", "manufacturing companies", "industrial finance" → annual_survey_of_industries
        Sample queries:
        Query1: "Number of factories manufacturing grain mill products in All India for fiscal year 2022-2023"
        Query2: "Depreciation to all industries in All India from April 2022 to March 2023"
        Query3: "Total bonus paid to all staff by industries in All India during FY 2022-23"
        Query4: "Wages and salaries in manufacturing sector in Maharashtra for 2021-22"
        Query5: "Gross value added by textile industry in Tamil Nadu"
        Query6: "Capital formation in steel industry across Indian states"
        Query7: "Employment in automotive manufacturing sector"
        Query8: "Working capital requirements for pharmaceutical industries"
        Query9: "Net profit of chemical industries in Gujarat"
        Query10: "Factory operations and production output in food processing"
        Query11: "Industrial establishments count by state and sector"
        Query12: "Materials consumed in cement manufacturing"
        Query13: "Fixed capital investment in power sector industries"
        Query14: "Electricity consumption by textile mills"
        Query15: "Total workers employed in mining industries"
        Query16: "Employers' contribution to provident fund in industries"
        Query17: "Interest paid by manufacturing companies"
        Query18: "Rent expenses for industrial facilities"
        Query19: "Stock of finished goods in automobile sector"
        Query20: "Total inputs cost for electronics manufacturing"

        28. asi_state_principal_characteristics: This table provides state-wise and sector-wise principal characteristics of industries in India, including employment and other key indicators, classified by NIC codes for the years 2021 to 2023.
        Instructions: Use this table to analyze industrial statistics such as number of workers, employees, and other principal characteristics by state, sector, year, and NIC classification. Filter by 'state', 'year', 'indicator', 'sector', or 'nic_code' as needed.
        Example queries:
        Query: Show the total number of workers in All India for the year 2022-23.
        Query: List the number of directly employed female workers by state for 2022-23.
        Query: Get the number of employees other than workers in Book Publishing for 2022-23.

        29. asi_imp_principal_characteristics_by_rural_urban_sector: This table provides annual data on principal industrial characteristics, such as the number of factories, split by rural and urban sectors in India.
        Instructions: Use this table to analyze trends or compare rural and urban industrial characteristics (like number of factories) across different years.
        Example queries:
        Query: Show the number of factories in rural and urban areas for each year.
        Query: List all principal characteristics available for 2012-13.
        Query: Get the total number of factories in rural areas between 2010-11 and 2012-13.

        30. asi_imp_principal_characteristics_india_by_mjr_indus_grp: This table provides annual data on the number of factories in India, categorized by major industry groups, as reported by the Enterprise Survey Division of MoSPI.
        Instructions: Use this table to analyze trends or compare the number of factories across different industries and years. Filter by 'year' for specific periods or by industry columns for sector-specific insights. The 'characteristics' column describes the metric (e.g., 'No. of Factories').
        Example queries:
        Query: Show the number of factories in the food products industry for each year.
        Query: Which year had the highest number of factories in the textiles sector?
        Query: List the number of factories for all industries in 2011-12.
        Query: Show the trend of factories in the pharmaceuticals industry over the years.

        31. asi_industrywise_factories_2022_23: This table provides the number of factories in India for various types of industries for the year 2022-23, along with metadata such as release date, last update, and data source.
        Instructions: Use this table to find the count of factories by industry type, or to analyze industry-wise distribution of factories for the year 2022-23.
        Example queries:
        Query: Show the number of factories for each industry type.
        Query: Which industry has the highest number of factories?
        Query: List all industries with more than 10,000 factories.
        Query: What is the total number of factories across all industries?
        Query: Show the data source and last updated date for the industry-wise factories data.

        32. asi_num_of_factories_nva: This table provides annual data on the number of factories and their net value added (in lakhs) as recorded by the Enterprise Survey Division, MoSPI.
        Instructions: Use this table to analyze trends in the number of factories and their net value added over different years. You can filter by year, aggregate values, or compare data across years.
        Example queries:
        Query: Show the net value added and number of factories for each year.
        Query: Find the year with the highest net value added.
        Query: Get the total net value added across all years.
        Query: List all years where the number of factories was below 100,000.

        33. asi_statewise_number_of_factories_for_2022_23: This table provides the number of factories in each Indian state for the year 2022-23, along with release and update dates and the data source.
        Instructions: Use this table to retrieve or analyze the count of factories by state for 2022-23, or to filter by release/update dates or data source.
        Example queries:
        Query: Show the number of factories in each state.
        Query: Which state had the highest number of factories in 2022-23?
        Query: List all states where the number of factories is greater than 30,000.
        Query: Give me the release date for the factory data.

        34. asi_top_ten_states_by_number_of_factories: This table shows the number of factories in the top ten Indian states for each year from 2010-11 to 2022-23, along with metadata about data release and source.
        Instructions: Use this table to analyze trends, compare, or retrieve the number of factories in leading Indian states across different years.
        Example queries:
        Query: Which state had the highest number of factories in 2022-23?
        Query: Show the number of factories in Maharashtra and Gujarat for the years 2010-11 and 2022-23.
        Query: List the trend of factories in Tamil Nadu from 2010-11 to 2022-23.
        Query: Which state saw the largest increase in the number of factories between 2010-11 and 2022-23?

        35. asi_trend_imp_characteristics_technical_coefficients: This table contains annual technical coefficients data, such as Fixed Capital to Output ratios, for different years, including metadata like release and update dates and data source.
        Instructions: Use this table to retrieve technical coefficients by year, region, or data source, or to analyze trends in industrial technical characteristics over time.
        Example queries:
        Query: Show the technical coefficients for all years for Fixed Capital to Output.
        Query: List all available regions in the table.
        Query: Get the technical coefficient for the year 2012-13.
        Query: Find the latest update date for the data.

        36. asi_trend_of_imp_characteristics_structural_ratios: This table provides annual data on key structural ratios, such as fixed capital per factory, across different regions and years, sourced from the Enterprise Survey Division MoSPI.
        Instructions: Use this table to analyze trends in structural ratios (e.g., fixed capital per factory) over time or by region. Filter by year, region, or data source as needed.
        Example queries:
        Query: Show the fixed capital per factory for each year.
        Query: List all available regions in the table.
        Query: Get the structural ratios for the year 2011-12.

        37. asi_trend_of_imp_principal_characteristics_india: This table contains annual data on key industrial characteristics in India, such as the number of factories, with associated metadata including release and update dates, and data source.
        Instructions: Use this table to retrieve historical trends and statistics related to principal industrial characteristics (e.g., number of factories) in India by year.
        Example queries:
        Query: Show the number of factories in India for each year.
        Query: Get all available data for the year 1982-83.
        Query: List the years and values for all records sourced from 'Enterprise Survey Division MoSPI'.
        
        
        38. none_of_these: for any queries which are unrelated to above files.

        ## Consider the list above, and respond ONLY with one of the file names from the following list:
        [asuse_est_annual_gva_per_establishment, asuse_est_num_establishments_pursuing_mixed_activity, asuse_per1000_estb_by_hours_worked_per_day, asuse_per1000_estb_by_months_operated_last_365days, asuse_per1000_estb_registered_under_acts_authorities, asuse_per1000_estb_using_computer_internet_last365_days, asuse_per1000_of_estb_using_internet_by_type_of_its_use, asuse_per1000_proppartn_estb_by_edu_owner_mjr_partner, asuse_per1000_proppartn_estb_by_other_econ_activities,
        asuse_per1000_proppartn_estb_by_socialgroup_owner, asuse_per_1000_distri_of_establishments_by_nature_of_operation, asuse_per_1000_distri_of_establishments_by_type_of_location, asuse_per_1000_distri_of_establishments_by_type_of_ownership, asuse_per_1000_of_establishments_which_are_npis_and_non_npis, asuse_statewise_est_num_of_estb_pursuing_mixed_activity, asuse_statewise_est_num_of_estb_serving_as_franchisee_outlet, asuse_statewise_estimated_annual_gva_per_establishment_rupees,
        asuse_statewise_per1000_distri_of_estb_by_nature_of_operation, asuse_statewise_per1000_distri_of_estb_by_type_of_location, asuse_statewise_per1000_distri_of_estb_by_type_of_ownership, asuse_statewise_per1000_estb_by_hours_worked_per_day, asuse_statewise_per1000_estb_by_month_num_operated_last365_day, asuse_statewise_per1000_estb_maintain_post_bank_saving_acc, asuse_statewise_per1000_estb_registered_diff_acts_authorities, asuse_statewise_per1000_estb_use_computer_internet_last365_day,
        asuse_statewise_per1000_proppart_estb_by_social_grp_mjr_prtner, annual_survey_of_industries, asi_state_principal_characteristics, asi_imp_principal_characteristics_by_rural_urban_sector, asi_imp_principal_characteristics_india_by_mjr_indus_grp, asi_industrywise_factories_2022_23, asi_num_of_factories_nva, asi_statewise_number_of_factories_for_2022_23, asi_top_ten_states_by_number_of_factories, asi_trend_imp_characteristics_technical_coefficients, asi_trend_of_imp_characteristics_structural_ratios,
        asi_trend_of_imp_principal_characteristics_india, none_of_these]

        Do not include any reasoning traces or other text apart from the file name selected from the above list.
            """)
    selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_worker_surveys(query):
    system_instruction=dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        1. asuse_est_annual_emoluments_per_hired_worker: This table provides annual, all-India-level data on estimated annual emoluments and number of hired workers by industry (e.g., Manufacture of Rubber and Plastics Products, Land Transport, Education, Accommodation), sector (Rural/Urban/Combined), establishment type (Own Account, Hired Worker), and formality status (Formal/Informal). Example entries include annual emoluments in 'Education' (Urban, 2022-23: Rs 265,897) and hired workers in 'Wholesale on a Fee or Contract Basis' (Combined, Informal, 2023-24: 153,601).
        Sample queries:
        a. What were the estimated annual emoluments per hired worker for Land Transport in rural areas for 2022-23?
        b. How many hired workers were there in the Manufacture of Tobacco Products industry in All India combined for 2021-22?

        2. asuse_est_num_workers_by_employment_gender: This table provides annual, all-India estimates of the number of workers by economic activity (e.g., Manufacturing of Motor Vehicles, Real Estate, Land Transport), nature of establishment (Own Account, Hired Worker, or All), sector (Rural, Urban, Combined), and gender (Male, Female, All). Data is published by the National Sample Survey Office (MoSPI). Sample entries include 'Manufacture of Beverages' (2023-24) and 'Other Retail Trade' (2022-23).
        Sample queries:
        a. How many female workers were engaged in land transport in urban areas during 2021-22?
        b. What was the estimated number of workers in 'Manufacture of Beverages' at all-India level for 2023-24?

        3. asuse_est_value_key_characteristics_by_workers: This table provides estimated values of key characteristics (such as input per worker) by number of workers, sector, activity category, and region for different years, based on data from the National Sample Survey Office.
        Instructions: Use this table to analyze or retrieve estimated values of key economic characteristics (like input per worker) segmented by year, state/UT, sector (rural/urban), broad activity category (e.g., Manufacturing), and number of workers.
        Example queries:
        Query: Show the input per worker for manufacturing in rural India for 2023-24, broken down by number of workers.
        Query: Get all key characteristics for urban sector in 2023-24 for the state of Maharashtra.
        Query: What is the estimated value for input per worker in the rural sector for all activity categories in 2023-24?

        4. asuse_estimated_annual_gva_per_worker_rupees: This table provides the estimated annual GVA (Gross Value Added) per worker (in rupees) for various activity categories, establishment types, and sectors across Indian states and union territories, based on data from the National Sample Survey Office.
        Instructions: Use this table to analyze or retrieve GVA (Gross Value Added) per worker statistics by year, state/UT, sector (rural/urban), activity category, and establishment type. Filter by these columns to get specific GVA (Gross Value Added) values or trends.
        Example queries:
        Query: What was the estimated annual GVA (Gross Value Added) per worker for 'Cotton Ginning, Cleaning and Bailing' in rural India in 2023-24 for Hired Worker Establishments?
        Query: Show the GVA (Gross Value Added) per worker for all establishment types in rural India for 'Cotton Ginning, Cleaning and Bailing' in 2023-24.
        Query: List the estimated annual GVA (Gross Value Added) per worker for each activity category in rural India for 2023-24.

        5. asuse_estimated_number_of_workers_by_type_of_workers: This table presents annual, all-India estimates of worker numbers by industry category (e.g., Water Transport, Manufacture of Textiles), worker type (e.g., Formal Hired Workers, Unpaid Family Member), gender, and establishment type (All, Hired Worker, Own Account). Data is disaggregated for urban, rural, and combined areas. Examples include 116 male informal water transport workers (urban, 2023-24) and 2,006 female unpaid family workers in manufacturing (rural, 2022-23).
        Sample queries:
        a. How many informal hired workers were there in urban food and accommodation service activities in 2023-24?
        b. What is the estimated number of female working owners in rural trading activities for 2021-22?

        6. asuse_statewise_est_num_of_worker_by_employment_and_gender: This table provides annual state/UT-wise estimates of the number of workers in India by industry (e.g., Manufacturing, Trade, Other Services), establishment type (Own Account, Hired Worker, All), gender, work status (Full/Part time), and location (Rural/Urban/Combined). Data covers multiple states such as Assam, Maharashtra, Kerala, and union territories like Chandigarh, for years like 2021-22 to 2023-24. Example: 182,703 male full-time manufacturing workers in urban Andhra Pradesh (2023-24).
        Sample queries:
        a. How many female hired workers were employed full-time in manufacturing establishments in Jammu and Kashmir for 2022-23?
        b. What is the estimated number of part-time male workers in trade sector own account establishments in rural West Bengal for 2023-24?

        7. asuse_statewise_estimated_annual_emoluments_per_hired_worker: This table presents annual, state/UT-wise data on estimated annual emoluments (in Rs.) and hired worker counts across India, sourced from the National Sample Survey Office (MoSPI). Data is available by sector (e.g., Manufacturing, Trade, Other Services), establishment type (Own Account, Hired Worker, All), formality (Formal/Informal/All), and area (Rural, Urban, Combined) for years 2021-22 to 2023-24. Sample entries include Uttar Pradesh, Maharashtra, Meghalaya, Telangana, and Delhi.
        Sample queries:
        a. What was the annual emolument per hired worker in the manufacturing sector for rural Karnataka in 2021-22?
        b. How many hired workers were estimated in all establishments of urban Maharashtra in 2021-22?

        8. asuse_statewise_estimated_annual_gva_per_worker_rupees: This table provides state and sector-wise estimated annual Gross Value Added GVA (Gross Value Added) per worker (in rupees) for different establishment types and broad activity categories, as reported by the National Sample Survey Office.
        Instructions: Use this table to retrieve GVA (Gross Value Added) per worker data by year, state/UT, sector (rural/urban), establishment type, and activity category. Filter using columns like 'year', 'stateut', 'sector', 'broad_activity_category', and 'establishment_type' as needed.
        Example queries:
        Query: Show the estimated annual GVA (Gross Value Added) per worker for all establishment types in Andhra Pradesh for 2023-24.
        Query: List the GVA (Gross Value Added) per worker in rural manufacturing sector for each state in 2023-24.
        Query: What is the GVA (Gross Value Added) per worker for Hired Worker Establishments in Andhra Pradesh's rural manufacturing sector in 2023-24?

        9. asuse_statewise_estimated_number_of_workers_by_type_of_workers: This table contains state/UT-wise and all-India annual data on the estimated number of workers by type (e.g., formal/informal hired workers, working owners, unpaid family members) in different sectors (Trade, Manufacturing, Other Services). Data is further broken down by urban/rural/combined, establishment type (All Establishments, Hired Worker, Own Account), and gender. Examples include Rajasthan (rural, Other Services, Other Workers) and Gujarat (urban, Trade, Total workers). Source: National Sample Survey Office, MoSPI.
        Sample queries:
        a. What is the number of female informal hired workers in rural Gujarat in the Trade sector for 2022-23?
        b. Show the annual estimated number of working owners in manufacturing establishments in Madhya Pradesh (urban) for the last three years.

        10. asi_no_of_workers_and_person_engaged: This table provides annual data on the number of workers and total persons engaged in enterprises, along with release and update information and the data source.
        Instructions: Use this table to retrieve historical statistics on workforce size and total engagement in enterprises for specific years or to analyze trends over time.
        Example queries:
        Query: Show the number of workers and total persons engaged for the year 1982-83.
        Query: List all years with their corresponding number of workers.
        Query: Find the year with the highest total persons engaged.

        11. periodic_labour_force_survey: This table contains Periodic Labour Force Survey (PLFS) data - the primary source for employment and unemployment statistics. Contains Labour Force Participation Rate (LFPR), Worker Population Ratio (WPR), Unemployment Rate (UR) by year, state, gender, age group, sector, religion, social group, education levels.
        Use this table for queries about: employment rates, unemployment statistics, labour force participation, worker demographics, job market analysis, employment by education/gender/age.

        **Query Classification Guidelines:**
        - If query mentions "labour force", "unemployment rate", "employment statistics", "job market", "worker participation", "employment demographics", "labour force participation", "workforce", "job seekers" → periodic_labour_force_survey
        Sample queries:
        Query1: "Labour force participation rate in India for 2022-23"
        Query2: "Unemployment rate by gender in urban areas"
        Query3: "Worker population ratio trends over last 5 years"
        Query4: "Employment statistics for graduates in rural areas"
        Query5: "Job market analysis for women in agriculture sector"
        Query6: "Labour force participation by education level"
        Query7: "Unemployment among youth aged 15-29 years"
        Query8: "Employment rates by social group and religion"
        Query9: "Worker demographics in services sector"
        Query10: "Labour market trends by state and region"
        Query11: "Employment opportunities for skilled workers"
        Query12: "Workforce participation in informal sector"
        Query13: "Employment statistics for different age groups"
        Query14: "Labour force survey data for metropolitan cities"
        Query15: "Job seekers and employment status analysis"

        12. lpfr_state_age: The 'lpfr_state_age' table provides Labour Force Participation Rate (LFPR) data by state, age group, gender, and area (rural/urban/total) for specified time periods.
        Instructions: Use this table to analyze LFPR statistics across Indian states, broken down by age group, gender, rural/urban/total populations, and time period. Filter by 'state', 'age_group', 'year', or area/gender columns as needed.
        Example queries:
        Query: Show the total LFPR for males and females in Andhra Pradesh for April-June 2025.
        Query: List rural and urban LFPR for persons aged 15-29 in Assam in 2025.
        Query: Get LFPR for all states for females in rural areas for 2025.

        13. cws_industry_distribution_state: This table provides the percentage distribution of workers by industry sector (agriculture, secondary including mining and quarrying, tertiary, and all sectors) across Indian states, segmented by age group, gender, area type, and time period.
        Instructions: Use this table to analyze the sectoral distribution of workers in different states, filtered by age group, gender, area type, and specific time periods. Useful for comparing employment patterns across sectors and demographics.
        Example queries:
        Query: Show the percentage of male workers in the agriculture sector in rural Bihar for April to June 2025.
        Query: List the sector-wise distribution for all sectors in Assam for rural males aged 15 years and above in Q2 2025.
        Query: Get the tertiary sector percentage for Andhra Pradesh for rural males aged 15 years and above in the latest available data.

        14. wpr_state_age: This table contains state-wise Worker Population Ratio (WPR) data by gender, area (rural/urban/total), and age group, along with time period, year, data source, and state information.
        Instructions: Use this table to analyze or retrieve WPR statistics by state, gender, area, age group, and time period. Filter by columns like 'state', 'year', 'age_group', or area/gender-specific WPR values as needed.
        Examples queries
        Query: Show the total WPR for males and females in Assam for the age group 15-29 years in 2025.
        Query: List all states with rural female WPR above 20 for April-June 2025.
        Query: Get the urban person WPR for Bihar for the latest released data.

        15. ur_state_age: This table provides state-wise unemployment rates segmented by age group, gender, and rural/urban areas, along with time period and data source details.
        Instructions: Use this table to analyze or retrieve unemployment rates by state, age group, gender, area (rural/urban/total), and time period (month, year). Filter by columns such as state, year, age_group, or area-specific rates as needed.
        Examples queries:
        Query: Show the total unemployment rate for persons aged 15-29 years in Andhra Pradesh for April to June 2025.
        Query: List rural female unemployment rates for all states in 2025.
        Query: Get urban male and female unemployment rates in Bihar for 2025.
        Query: Find the data source and release date for Assam's unemployment data for April to June 2025.

        16. epfo_india_mth_payroll: This table contains monthly and yearly payroll data from EPFO India, showing the number of new subscribers by age group, total new subscribers, and the number of establishments remitting their first ECR.
        Instructions: Use this table to analyze trends in EPFO payroll enrollments by age group, month, or year, and to examine the number of new establishments joining the EPFO scheme.
        Example queries:
        Query: Show the total number of new EPFO subscribers for each year.
        Query: Get the number of new subscribers aged 18-21 for each month.
        Query: Find the month with the highest number of establishments remitting their first ECR.
        Query: Show the total new subscribers by age group for the year 2019-20.
        
        17. epfo_nat_ind_exempted_establishments_list : This table contains a list of exempted establishments under EPFO, including their IDs, names, the relevant month, and release/update dates.
        Instructions: Use this table to find information about exempted establishments, such as their names, IDs, and the months for which they are listed. You can filter by establishment name, ID, month, or date fields.
        Example queries:
        User Query: Show all exempted establishments for August 2024.
        User Query: List the names and IDs of establishments updated after October 2025.
        User Query: Find the release date for 'ROURKELA STEEL PLANT'.

        18. none_of_these: for any queries which are unrelated to above files.

        ## Consider the list above, and respond ONLY with one of the file names from the following list:
        [asuse_est_annual_emoluments_per_hired_worker,asuse_est_num_workers_by_employment_gender, asuse_est_value_key_characteristics_by_workers, asuse_estimated_annual_gva_per_worker_rupees, asuse_estimated_number_of_workers_by_type_of_workers, asuse_statewise_est_num_of_worker_by_employment_and_gender, asuse_statewise_estimated_annual_emoluments_per_hired_worker, aasuse_statewise_estimated_annual_gva_per_worker_rupees,
        asuse_statewise_estimated_number_of_workers_by_type_of_workers, asi_no_of_workers_and_person_engaged,periodic_labour_force_survey, lpfr_state_age, cws_industry_distribution_state, wpr_state_age, ur_state_age, epfo_india_mth_payroll,epfo_nat_ind_exempted_establishments_list,  none_of_these]

        Do not include any reasoning traces or other text apart from the file name selected from the above list.
            """)
    selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

# def file_selector_enterprise_establishment_surveys(query):
#     system_instruction=dedent(f"""
#         You are tasked with identifying the file that contains the required data based on the query: "{query}".
#         You must pick one file name only from the following list:

#         Choose a file only if the table description explicitly confirms that the data required by the query is covered.

#         1. asuse_est_annual_emoluments_per_hired_worker: This table provides annual, all-India-level data on estimated annual emoluments and number of hired workers by industry (e.g., Manufacture of Rubber and Plastics Products, Land Transport, Education, Accommodation), sector (Rural/Urban/Combined), establishment type (Own Account, Hired Worker), and formality status (Formal/Informal). Example entries include annual emoluments in 'Education' (Urban, 2022-23: Rs 265,897) and hired workers in 'Wholesale on a Fee or Contract Basis' (Combined, Informal, 2023-24: 153,601).
#         Sample queries:
#         a. What were the estimated annual emoluments per hired worker for Land Transport in rural areas for 2022-23?
#         b. How many hired workers were there in the Manufacture of Tobacco Products industry in All India combined for 2021-22?


#         2. asuse_est_annual_gva_per_establishment: This table provides estimated annual Gross Value Added (GVA (Gross Value Added)) per establishment, categorized by year, state/UT, sector, activity category, and establishment type, sourced from the National Sample Survey Office.
#         Instructions: Use this table to analyze or compare the estimated annual GVA (Gross Value Added) per establishment across different years, states/UTs, sectors (rural/urban), activity categories, and establishment types (such as Hired Worker Establishments or Own Account Establishments). Filter by relevant columns to get specific insights.
#         Example queries:
#         Query: What was the estimated annual GVA (Gross Value Added) per establishment for Cotton Ginning, Cleaning and Bailing in rural All India for 2023-24?
#         Query: Show the GVA (Gross Value Added) per establishment for all establishment types in 2023-24 for rural sector.
#         Query: List the years and GVA (Gross Value Added) per establishment for Hired Worker Establishments in Cotton Ginning, Cleaning and Bailing activity.

#         3. asuse_est_num_establishments_pursuing_mixed_activity: This table presents annual, all-India estimates of establishments pursuing mixed activities, disaggregated by sector (urban, rural, combined), establishment type (Own Account, Hired Worker, All), and economic activity (e.g., Tobacco Products, Professional Services, Real Estate, Food Manufacturing). Data come from the National Sample Survey Office, MoSPI. Examples: 452,291 urban OAEs in tobacco manufacture (2022-23), 10,336,002 rural retail trade establishments (2023-24), and 24,643,235 all-India service providers (2022-23).
#         Sample queries:
#         a. How many rural own account establishments were engaged in real estate activities in 2022-23?
#         b. What is the estimated number of establishments involved in 'Other Retail Trade' at all-India level for 2023-24?


#         4. asuse_est_num_workers_by_employment_gender: This table provides annual, all-India estimates of the number of workers by economic activity (e.g., Manufacturing of Motor Vehicles, Real Estate, Land Transport), nature of establishment (Own Account, Hired Worker, or All), sector (Rural, Urban, Combined), and gender (Male, Female, All). Data is published by the National Sample Survey Office (MoSPI). Sample entries include 'Manufacture of Beverages' (2023-24) and 'Other Retail Trade' (2022-23).
#         Sample queries:
#         a. How many female workers were engaged in land transport in urban areas during 2021-22?
#         b. What was the estimated number of workers in 'Manufacture of Beverages' at all-India level for 2023-24?


#         5. asuse_est_value_key_characteristics_by_workers: This table provides estimated values of key characteristics (such as input per worker) by number of workers, sector, activity category, and region for different years, based on data from the National Sample Survey Office.
#         Instructions: Use this table to analyze or retrieve estimated values of key economic characteristics (like input per worker) segmented by year, state/UT, sector (rural/urban), broad activity category (e.g., Manufacturing), and number of workers.
#         Example queries:
#         Query: Show the input per worker for manufacturing in rural India for 2023-24, broken down by number of workers.
#         Query: Get all key characteristics for urban sector in 2023-24 for the state of Maharashtra.
#         Query: What is the estimated value for input per worker in the rural sector for all activity categories in 2023-24?

#         6. asuse_estimated_annual_gva_per_worker_rupees: This table provides the estimated annual Gross Value Added (GVA (Gross Value Added)) per worker (in rupees) for various activity categories, establishment types, and sectors across Indian states and union territories, based on data from the National Sample Survey Office.
#         Instructions: Use this table to analyze or retrieve GVA (Gross Value Added) per worker statistics by year, state/UT, sector (rural/urban), activity category, and establishment type. Filter by these columns to get specific GVA (Gross Value Added) values or trends.
#         Example queries:
#         Query: What was the estimated annual GVA (Gross Value Added) per worker for 'Cotton Ginning, Cleaning and Bailing' in rural India in 2023-24 for Hired Worker Establishments?
#         Query: Show the GVA (Gross Value Added) per worker for all establishment types in rural India for 'Cotton Ginning, Cleaning and Bailing' in 2023-24.
#         Query: List the estimated annual GVA (Gross Value Added) per worker for each activity category in rural India for 2023-24.


#         7. asuse_estimated_number_of_workers_by_type_of_workers: This table presents annual, all-India estimates of worker numbers by industry category (e.g., Water Transport, Manufacture of Textiles), worker type (e.g., Formal Hired Workers, Unpaid Family Member), gender, and establishment type (All, Hired Worker, Own Account). Data is disaggregated for urban, rural, and combined areas. Examples include 116 male informal water transport workers (urban, 2023-24) and 2,006 female unpaid family workers in manufacturing (rural, 2022-23).
#         Sample queries:
#         a. How many informal hired workers were there in urban food and accommodation service activities in 2023-24?
#         b. What is the estimated number of female working owners in rural trading activities for 2021-22?


#         8. asuse_per1000_estb_by_hours_worked_per_day: This table reports annual, All-India level data on the per-1000 distribution of establishments by hours worked per day, categorized by rural/urban/combined sectors and establishment types such as Own Account Establishments and Hired Worker Establishments. Industry categories include 'Manufacture of Beverages', 'Trading Activities', 'Financial Services', and 'Food and Accommodation Service Activities', with working hours grouped as '<4', '4-7', '8-11', '>11', and 'All'. Example: 705 establishments (urban, professional activities) work '8-11' hours.
#         Sample queries:
#         a. What percentage of rural establishments in the Manufacture of Beverages category worked more than 11 hours a day in 2023-24?
#         b. How does the distribution of working hours differ between urban and rural establishments in the Financial Service Activities Except Insurance and Pension Funding sector in 2022-23?


#         9. asuse_per1000_estb_by_months_operated_last_365days: This table presents annual all-India data on the distribution of establishments by months of operation, disaggregated by sector (e.g., Manufacture of Electrical Equipment, Real Estate Activities, Water Transport), type (Own Account, Hired Worker), and rural/urban/combined geographies. Key metrics include per-1000 distribution or average months operated for categories such as '<= 3 Months', '7 to 9 Months', and '> 9 Months'. Data is sourced from the National Sample Survey Office (MoSPI).
#         Sample queries:
#         a. What was the average number of months operated by rural establishments engaged in the manufacture of leather and related products in 2023-24?
#         b. How many per 1000 urban hired worker establishments in wholesale on a fee or contract basis operated for less than or equal to 3 months in 2021-22?


#         10. asuse_per1000_estb_registered_under_acts_authorities: This table provides annual, all-India data on the number per 1000 of establishments registered under various Acts or authorities (e.g., Societies Reg. Act, CGST Act, EPFO/ESIC, RTO). The data is category-wise (urban, rural, combined), sector-wise (e.g., Manufacture of Textiles, Information and Communication, Trading Activities), and by type of establishment (All/Own Account/Hired Worker). For example, in 2021-22, 277 per 1000 urban hired worker establishments in wood manufacturing were registered under Shops & Establishments Act.
#         Sample queries:
#         a. What was the number per 1000 of rural establishments registered under the CGST Act for 'Wholesale and Retail Trade of Motor Vehicles and Motor Cycles' in 2021-22?
#         b. How did registration of hired worker establishments in 'Non-captive Electricity Generation and Transmission' change between 2022-23 and 2023-24 under 'Others' in urban areas?


#         11. asuse_per1000_estb_using_computer_internet_last365_days: This table presents annual, all-India data from the National Sample Survey Office on the number per 1000 establishments using computers and internet during the last 365 days. The data is available by sector (Trade, Manufacturing, Other Services), location (Rural, Urban, Combined), and type (Own Account Establishments, Hired Worker Establishments, All). For example, in 2023-24, 129 per 1000 urban service establishments used computers, while 640 per 1000 hired worker trade establishments used internet.
#         Sample queries:
#         a. What was the number per 1000 manufacturing establishments using the internet in urban areas in 2023-24?
#         b. How many per 1000 own account trade establishments in rural areas used computers in 2022-23?


#         12. asuse_per1000_of_estb_using_internet_by_type_of_its_use: This table provides annual, all-India statistics on the number per 1,000 of establishments using the Internet, broken down by sector (e.g., Manufacturing, Trade, Other Services), area (Rural, Urban, Combined), and specific types of Internet use (such as Internet Banking, Delivering Products Online, Telephoning Over VoIP, Customer Services). Example entries include: Manufacturing establishments in rural India using the Internet for government information in 2022-23 (46), or Urban Trade sector using it for staff training in 2021-22 (323).
#         Sample queries:
#         a. How many establishments per 1,000 in the Trade sector used Internet banking in urban India in 2021-22?
#         b. What was the number per 1,000 of rural manufacturing establishments using the Internet for accessing financial services in 2023-24?


#         13. asuse_per1000_proppartn_estb_by_edu_owner_mjr_partner: This dataset provides the annual (financial year) per 1000 distribution of proprietary and partnership establishments in India, categorized by the level of general education of the owner or major partner. Data is available at the all-India level, disaggregated by rural, urban, and combined sectors. Education categories include Not Literate, Literate Below Primary, Literate Graduate and Above, among others. Example: 199 urban establishments per 1000 had graduate owners in 2022-23.
#         Sample queries:
#         a. What percentage of rural proprietary establishments in 2022-23 were owned by people with primary to below secondary education?
#         b. How did the distribution of urban establishments owned by graduates change between 2021-22 and 2023-24?


#         14. asuse_per1000_proppartn_estb_by_other_econ_activities: This table provides annual, all-India level data on the distribution per 1,000 Proprietary and Partnership establishments by various economic activities (e.g., Manufacturing Activities, Food and Accommodation Service Activities, Real Estate, Education). Data is disaggregated by rural/urban/combined sectors, establishment size (e.g., Hired Worker Establishments, Own Account Establishments), and the number of other economic activities present. Data spans years like 2021-22 to 2023-24. Source: National Sample Survey Office, MoSPI.
#         Sample queries:
#         a. What was the per 1000 distribution of proprietary and partnership establishments for Manufacture of Pharmaceuticals in urban India in 2022-23?
#         b. Show the annual trend from 2021-22 to 2023-24 for All Establishments in Food and Accommodation Service Activities at the all-India level.


#         15. asuse_per1000_proppartn_estb_by_socialgroup_owner: This table presents all-India level, annual data on the per 1000 distribution of proprietary and partnership establishments by the social group of owner or major partner. Data is provided across urban, rural, and combined sectors, for various establishment types (Own Account, Hired Worker, All), and industries such as Manufacture of Rubber and Plastics Products, Accommodation, and Trading Activities. Social group categories include Scheduled Tribe, Scheduled Caste, OBC, Others, and Not Known.
#         Sample queries:
#         a. What is the distribution of hired worker establishments owned by Scheduled Tribe groups in the financial service sector for 2022-23?
#         b. How do the per 1000 establishment distributions for the manufacture of paper and paper products differ between urban and rural areas for OAE in 2023-24?


#         16. asuse_per_1000_distri_of_establishments_by_nature_of_operation: This table presents annual all-India estimates of the per 1000 distribution of establishments by nature of operation, disaggregated by sector (e.g., manufacture of food products, education, retail trade), type of establishment (Own Account, Hired Worker, All), and operational status (Perennial, Seasonal, Casual), for rural, urban, and combined regions. Data examples include 'Manufacture of Furniture' (urban, all), 'Other Wholesale Trade' (rural, HWE, seasonal), and 'Human Health and Social Work Activities'.
#         Sample queries:
#         a. What percentage of urban establishments in India engaged in manufacture of furniture are perennial according to the latest available year?
#         b. How does the distribution of hired worker establishments in the education sector vary between rural and urban areas for 2022-23?


#         17. asuse_per_1000_distri_of_establishments_by_type_of_location: This table presents annual all-India data on the per 1000 distribution of establishments by type of location, categorized by rural, urban, and combined regions. Industries covered range from 'Other Manufacturing', 'Land Transport', 'Education', 'Financial Service Activities' to 'Manufacture of Chemicals'. Data is further detailed for own account, hired worker, and all establishments across location types such as household premises, permanent and temporary structures, and mobile markets.
#         Sample queries:
#         a. What is the distribution of 'Land Transport' establishments located outside household premises with permanent structure in rural areas for 2021-22?
#         b. How many 'Manufacture of Chemicals and Chemical Products' own account establishments operated within household premises in urban India in 2023-24 per 1000 units?


#         18. asuse_per_1000_distri_of_establishments_by_type_of_ownership: This table presents all-India annual data on the per-1000 distribution of establishments by ownership type, for different activity categories (e.g., Manufacture of Textiles, Trading Activities, Accommodation) and sectors (urban, rural, combined). Ownership types include Proprietary-Male, SHG, Partnership, Co-operatives, among others. For example, in 2023-24, 908 per 1000 urban 'Other Financial Activities' establishments were 'Proprietary-Male', and 1000 per 1000 urban 'Accommodation' HWE were 'All'.
#         Sample queries:
#         a. What percentage of urban establishments under 'Manufacture of Tobacco Products' are owned by women in 2023-24?
#         b. How does the distribution of ownership type in 'Accommodation' activities differ between rural and urban India for 2022-23?


#         19. asuse_per_1000_of_establishments_which_are_npis_and_non_npis: This table presents annual state-wise data on the number per 1000 of establishments classified as NPIs (Non-Profit Institutions) and non-NPIs, disaggregated by urban, rural, and combined sectors across India. Categories include 'Trade', 'Manufacturing', and 'Other Services', with receipt sources such as 'Donation/Grants' and 'Other Sources'. Examples include 947 non-NPIs per 1000 establishments in urban Nagaland (2023-24) and all-India combined data for manufacturing in 2022-23.
#         Sample queries:
#         a. What percentage of trade establishments were NPIs with major receipts from donations in Rajasthan (urban) in 2021-22?
#         b. Show state-wise data for non-NPI establishments in other services for the year 2022-23.


#         20. asuse_statewise_est_num_of_estb_pursuing_mixed_activity: This table provides annual state/UT-wise data on the estimated number of establishments pursuing mixed activities in India, disaggregated by sector (e.g., Manufacturing, Trade, Other Services), area (Urban, Rural, Combined), and establishment type (Own Account Establishments, Hired Worker Establishments, All). Data examples include 763,860 rural manufacturing establishments in Maharashtra (2022-23), 9,483 urban HWE in Chandigarh (2022-23), and 2,197,497 combined OAE in Odisha (2021-22).
#         Sample queries:
#         a. How many urban hired worker establishments pursuing mixed activities were there in Uttarakhand in 2023-24?
#         b. Provide the number of manufacturing establishments in Tamil Nadu (all establishment types) for 2022-23, broken down by area.


#         21. asuse_statewise_est_num_of_estb_serving_as_franchisee_outlet: This table contains annual state/UT-wise estimates of the number of establishments serving as franchisee outlets across India from 2021-22 to 2023-24, with data provided at both state (e.g., Bihar: 1129; Odisha: 5826; Telangana: 7308) and all-India levels (e.g., 132471 in 2023-24). Data covers all major states and union territories, including minor entries (e.g., Lakshadweep: 0), and is sourced from the National Sample Survey Office, MoSPI.
#         Sample queries:
#         a. How many franchisee outlets were estimated in Tamil Nadu in each of the last three years?
#         b. Which Indian state had the highest number of franchisee establishments in 2022-23 according to the NSSO?


#         22. asuse_statewise_est_num_of_worker_by_employment_and_gender: This table provides annual state/UT-wise estimates of the number of workers in India by industry (e.g., Manufacturing, Trade, Other Services), establishment type (Own Account, Hired Worker, All), gender, work status (Full/Part time), and location (Rural/Urban/Combined). Data covers multiple states such as Assam, Maharashtra, Kerala, and union territories like Chandigarh, for years like 2021-22 to 2023-24. Example: 182,703 male full-time manufacturing workers in urban Andhra Pradesh (2023-24).
#         Sample queries:
#         a. How many female hired workers were employed full-time in manufacturing establishments in Jammu and Kashmir for 2022-23?
#         b. What is the estimated number of part-time male workers in trade sector own account establishments in rural West Bengal for 2023-24?


#         23. asuse_statewise_estimated_annual_emoluments_per_hired_worker: This table presents annual, state/UT-wise data on estimated annual emoluments (in Rs.) and hired worker counts across India, sourced from the National Sample Survey Office (MoSPI). Data is available by sector (e.g., Manufacturing, Trade, Other Services), establishment type (Own Account, Hired Worker, All), formality (Formal/Informal/All), and area (Rural, Urban, Combined) for years 2021-22 to 2023-24. Sample entries include Uttar Pradesh, Maharashtra, Meghalaya, Telangana, and Delhi.
#         Sample queries:
#         a. What was the annual emolument per hired worker in the manufacturing sector for rural Karnataka in 2021-22?
#         b. How many hired workers were estimated in all establishments of urban Maharashtra in 2021-22?


#         24. asuse_statewise_estimated_annual_gva_per_establishment_rupees: This table provides state and sector-wise estimated annual Gross Value Added (GVA (Gross Value Added)) per establishment (in Rupees) for different establishment types and broad activity categories, as reported by the National Sample Survey Office.
#         Instructions: Use this table to analyze or retrieve estimated annual GVA (Gross Value Added) per establishment by state/UT, year, sector (rural/urban), establishment type (HWE/OAE/All), and broad activity category (e.g., Manufacturing). Filter by relevant columns to get specific GVA (Gross Value Added) values.
#         Example queries:
#         Query: Show the estimated annual GVA (Gross Value Added) per establishment for all establishment types in Andhra Pradesh (rural, manufacturing) for 2023-24.
#         Query: List the GVA (Gross Value Added) per establishment for each state in the manufacturing sector for 2023-24 (all establishment types, rural only).
#         Query: Get the GVA (Gross Value Added) per establishment for Own Account Establishments in Andhra Pradesh for 2023-24, manufacturing sector, rural area.


#         25. asuse_statewise_estimated_annual_gva_per_worker_rupees: This table provides state and sector-wise estimated annual Gross Value Added (GVA (Gross Value Added)) per worker (in rupees) for different establishment types and broad activity categories, as reported by the National Sample Survey Office.
#         Instructions: Use this table to retrieve GVA (Gross Value Added) per worker data by year, state/UT, sector (rural/urban), establishment type, and activity category. Filter using columns like 'year', 'stateut', 'sector', 'broad_activity_category', and 'establishment_type' as needed.
#         Example queries:
#         Query: Show the estimated annual GVA (Gross Value Added) per worker for all establishment types in Andhra Pradesh for 2023-24.
#         Query: List the GVA (Gross Value Added) per worker in rural manufacturing sector for each state in 2023-24.
#         Query: What is the GVA (Gross Value Added) per worker for Hired Worker Establishments in Andhra Pradesh's rural manufacturing sector in 2023-24?


#         26. asuse_statewise_estimated_number_of_workers_by_type_of_workers: This table contains state/UT-wise and all-India annual data on the estimated number of workers by type (e.g., formal/informal hired workers, working owners, unpaid family members) in different sectors (Trade, Manufacturing, Other Services). Data is further broken down by urban/rural/combined, establishment type (All Establishments, Hired Worker, Own Account), and gender. Examples include Rajasthan (rural, Other Services, Other Workers) and Gujarat (urban, Trade, Total workers). Source: National Sample Survey Office, MoSPI.
#         Sample queries:
#         a. What is the number of female informal hired workers in rural Gujarat in the Trade sector for 2022-23?
#         b. Show the annual estimated number of working owners in manufacturing establishments in Madhya Pradesh (urban) for the last three years.


#         27. asuse_statewise_per1000_distri_of_estb_by_nature_of_operation: This table provides state/UT-wise, rural/urban/combined, and all-India level annual data on the per 1000 distribution of establishments by nature of operation (perennial, seasonal, casual). Categories covered include Manufacturing, Trade, Other Services, and All, with breakdowns for Own Account Establishments (OAE), Hired Worker Establishments (HWE), and all establishments. Example entries: Uttar Pradesh, Gujarat, Sikkim, Puduchery, Maharashtra (2021–2024, all modes and categories).
#         Sample queries:
#         a. What percentage of manufacturing establishments in rural Uttar Pradesh were perennial in 2021-22?
#         b. Show the state-wise distribution of seasonal own account establishments in the trade sector for 2022-23.


#         28. asuse_statewise_per1000_distri_of_estb_by_type_of_location: This table presents annual, state/UT-wise data on the per-1000 distribution of establishments in India, categorized by sector (Manufacturing, Trade, Other Services), establishment type (Own Account Establishments, Hired Worker Establishments, All), and location type (e.g., within household premises, street vendors, kiosks). Coverage includes states like Tamil Nadu, Gujarat, West Bengal, and all-India aggregates, with data available separately for rural, urban, and combined geographies from 2021-22 to 2023-24.
#         Sample queries:
#         a. What is the distribution of manufacturing establishments by location type in rural West Bengal for 2023-24?
#         b. How does the per-1000 share of street vendor establishments in urban Gujarat compare between 2021-22 and 2023-24?


#         29. asuse_statewise_per1000_distri_of_estb_by_type_of_ownership: This table provides annual, state/UT-wise and all-India data on the per-1000 distribution of establishments by type of ownership, segmented by rural, urban, and combined areas. Categories include Manufacturing, Trade, and Other Services, with detailed breakdowns such as Proprietary-Male, Partnerships, Societies, and SHGs. Examples include Assam (Rural, Trade), Punjab (Urban, Other Services), and all-India rural data for Trade. Source: National Sample Survey Office, MoSPI.
#         Sample queries:
#         a. What is the share of proprietary-female owned establishments in the 'Other Services' sector in urban Punjab for 2023-24?
#         b. Show per 1000 distribution of SHG-owned establishments for all establishment types in rural Kerala for 2022-23.


#         30. asuse_statewise_per1000_estb_by_hours_worked_per_day: This table provides annual, state/UT-wise data on the per-1000 distribution of establishments by the number of hours normally worked per day. It covers various states (e.g., Goa, Maharashtra, Gujarat), sectors (Trade, Manufacturing, Other Services), establishment types (Own Account, Hired Worker, All Establishments), and urban/rural status. Examples include 813 Own Account Trade establishments in Goa and 815 Hired Worker Trade establishments in Assam (2023-24).
#         Sample queries:
#         a. What is the distribution of manufacturing establishments by hours worked in a day for Telangana (urban) in 2023-24?
#         b. Compare the proportion of Own Account versus Hired Worker establishments working 8-11 hours in Gujarat during 2022-23.


#         31. asuse_statewise_per1000_estb_by_month_num_operated_last365_day: This table presents annual data on the distribution per 1000 of establishments by number of months operated in the last 365 days, covering different states (e.g., Odisha, Karnataka, Delhi), sectors (Trade, Manufacturing, Other Services), and establishment types (Own Account, Hired Worker, All). Data is available at all-India, state, and urban/rural/combined levels with examples like Odisha (Rural, Trade, <=3 Months) and Delhi (Combined, Manufacturing, <=3 Months).
#         Sample queries:
#         a. How many own account establishments in Haryana traded for <=3 months during 2023-24?
#         b. Which state had the highest per 1000 distribution of trade establishments operating more than 9 months in rural areas in 2022-23?


#         32. asuse_statewise_per1000_estb_maintain_post_bank_saving_acc: This table presents annual, state/UT-wise and location-wise (urban/rural/combined) data on the number per 1000 establishments maintaining bank or post office savings accounts in India, split by sectors such as Trade, Manufacturing, and Other Services. Data is categorized for Own Account and Hired Worker Establishments, and by account holder type. Examples include Maharashtra-rural (317, HWE), Assam-urban (914, OAE), and Chandigarh-rural (1000, HWE), from 2021-22 to 2023-24.
#         Sample queries:
#         a. What was the number per 1000 of urban establishments in Karnataka maintaining any bank account in 2021-22?
#         b. How does the proportion of rural own account establishments with Post Office Savings Bank accounts in Punjab compare to Haryana in 2023-24?


#         33. asuse_statewise_per1000_estb_registered_diff_acts_authorities: This table provides annual, state/UT-wise and all-India statistics on the number per 1000 of establishments registered under various Acts and authorities (e.g., Shops & Establishment Act, EPFO/ESIC, Co-operative Societies Act) by sector (Trade, Manufacturing, Other Services) and type (Own Account, Hired Worker, All). Examples include 0 per 1000 for 'EPFO/ESIC' in Goa Rural (2021-22) and 148 per 1000 for 'Others' in All India Rural (2022-23).
#         Sample queries:
#         a. What is the number per 1000 of establishments registered under the Shops & Establishment Act in urban Odisha for 2021-22?
#         b. Provide state-wise data for 2023-24 on establishments registered under the Co-operative Societies Act, 1912 in the 'Other Services' sector.


#         34. asuse_statewise_per1000_estb_use_computer_internet_last365_day: This table presents annual, state/UT-wise data on the number per 1000 establishments using computers and internet in India, across years like 2021-22 and 2023-24. Data is disaggregated by sector (e.g., Trade, Manufacturing, Other Services), type of establishment (Own Account, Hired Worker, All), and area (Urban, Rural, Combined). Examples include Kerala (rural, Hired Worker, computer: 278), Chandigarh (Combined, Hired Worker, internet: 958), and Uttar Pradesh (urban, all, internet: 158).
#         Sample queries:
#         a. What is the number per 1000 of trade establishments using computers in urban Sikkim for 2023-24?
#         b. How many own account establishments in rural Madhya Pradesh used the internet per 1000 in 2022-23?


#         35. asuse_statewise_per1000_proppart_estb_by_social_grp_mjr_prtner: This table provides annual state/UT-wise and national-level data on the per 1000 distribution of proprietary and partnership establishments in India by the social group of the owner-major partner. It covers rural, urban, and combined geographies, establishment types (Own Account, Hired Worker, All), sectors like Manufacturing, Trade, and Other Services, and social groups including Scheduled Castes, Scheduled Tribes, Other Backward Classes, and Others. Example entries: Rajasthan, 2021-22, Manufacturing, Scheduled Caste; All India, 2023-24, Manufacturing, OBC.
#         Sample queries:
#         a. What is the proportion of Scheduled Tribe-owned manufacturing establishments in Meghalaya (rural) in 2023-24?
#         b. How did the distribution of Other Backward Classes in trade sector establishments change in Andhra Pradesh from 2021-22 to 2023-24?

#         36. annual_survey_of_industries: This table contains Annual Survey of Industries (ASI) data with comprehensive industrial performance metrics for India.
#         It includes:
#         **Financial Metrics**: Addition in Stock (Materials, Finished Goods, Semi-Finished Goods), Depreciation, Gross/Net Capital Formation, Gross/Net Value Added, Interest Paid/Received, Invested Capital, Net Income, Net Profit, Outstanding Loan, Physical Working Capital, Rent Paid/Received, Total Inputs/Output, Working Capital
#         **Employment & Labor**: Bonus to All Staff, Employers' Contribution, No. of Directly Employed Workers (Male/Female), No. of Employees Other Than Workers, No. of Workers Employed Through Contractors, Total Mandays Employed, Total Number of Persons Engaged, Wages and Salaries (with various breakdowns)
#         **Production & Operations**: Factories in Operation, Number of Factories, Fuels Consumed (Coal, Electricity, Petroleum, Other), Materials Consumed, Value of Product and By-Product
#         **Infrastructure**: Fixed Capital, Gross Value of Plant & Machinery, Quantity of Coal/Electricity Consumed
#         Use this table for queries about: industrial establishments, factory data, manufacturing statistics, capital investment, employment in industries, wages, production output, industrial finance, depreciation, bonus payments, factory operations, industrial sectors, manufacturing performance.

#         **Query Classification Guidelines:**
#         - If query mentions "industries", "factories", "manufacturing", "industrial sectors", "production", "depreciation", "bonus", "wages in industries", "capital formation", "industrial employment", "manufacturing companies", "industrial finance" → annual_survey_of_industries
#         Sample queries:
#         Query1: "Number of factories manufacturing grain mill products in All India for fiscal year 2022-2023"
#         Query2: "Depreciation to all industries in All India from April 2022 to March 2023"
#         Query3: "Total bonus paid to all staff by industries in All India during FY 2022-23"
#         Query4: "Wages and salaries in manufacturing sector in Maharashtra for 2021-22"
#         Query5: "Gross value added by textile industry in Tamil Nadu"
#         Query6: "Capital formation in steel industry across Indian states"
#         Query7: "Employment in automotive manufacturing sector"
#         Query8: "Working capital requirements for pharmaceutical industries"
#         Query9: "Net profit of chemical industries in Gujarat"
#         Query10: "Factory operations and production output in food processing"
#         Query11: "Industrial establishments count by state and sector"
#         Query12: "Materials consumed in cement manufacturing"
#         Query13: "Fixed capital investment in power sector industries"
#         Query14: "Electricity consumption by textile mills"
#         Query15: "Total workers employed in mining industries"
#         Query16: "Employers' contribution to provident fund in industries"
#         Query17: "Interest paid by manufacturing companies"
#         Query18: "Rent expenses for industrial facilities"
#         Query19: "Stock of finished goods in automobile sector"
#         Query20: "Total inputs cost for electronics manufacturing"

#         37. periodic_labour_force_survey: This table contains Periodic Labour Force Survey (PLFS) data - the primary source for employment and unemployment statistics. Contains Labour Force Participation Rate (LFPR), Worker Population Ratio (WPR), Unemployment Rate (UR) by year, state, gender, age group, sector, religion, social group, education levels.
#         Use this table for queries about: employment rates, unemployment statistics, labour force participation, worker demographics, job market analysis, employment by education/gender/age.

#         **Query Classification Guidelines:**
#         - If query mentions "labour force", "unemployment rate", "employment statistics", "job market", "worker participation", "employment demographics", "labour force participation", "workforce", "job seekers" → periodic_labour_force_survey
#         Sample queries:
#         Query1: "Labour force participation rate in India for 2022-23"
#         Query2: "Unemployment rate by gender in urban areas"
#         Query3: "Worker population ratio trends over last 5 years"
#         Query4: "Employment statistics for graduates in rural areas"
#         Query5: "Job market analysis for women in agriculture sector"
#         Query6: "Labour force participation by education level"
#         Query7: "Unemployment among youth aged 15-29 years"
#         Query8: "Employment rates by social group and religion"
#         Query9: "Worker demographics in services sector"
#         Query10: "Labour market trends by state and region"
#         Query11: "Employment opportunities for skilled workers"
#         Query12: "Workforce participation in informal sector"
#         Query13: "Employment statistics for different age groups"
#         Query14: "Labour force survey data for metropolitan cities"
#         Query15: "Job seekers and employment status analysis"

#         38. asi_state_principal_characteristics: This table provides state-wise and sector-wise principal characteristics of industries in India, including employment and other key indicators, classified by NIC codes for the years 2021 to 2023.
#         Instructions: Use this table to analyze industrial statistics such as number of workers, employees, and other principal characteristics by state, sector, year, and NIC classification. Filter by 'state', 'year', 'indicator', 'sector', or 'nic_code' as needed.
#         Example queries:
#         Query: Show the total number of workers in All India for the year 2022-23.
#         Query: List the number of directly employed female workers by state for 2022-23.
#         Query: Get the number of employees other than workers in Book Publishing for 2022-23.

#         39. epfo_india_mth_payroll: This table contains monthly and yearly payroll data from EPFO India, showing the number of new subscribers by age group, total new subscribers, and the number of establishments remitting their first ECR.
#         Instructions: Use this table to analyze trends in EPFO payroll enrollments by age group, month, or year, and to examine the number of new establishments joining the EPFO scheme.
#         Example queries:
#         Query: Show the total number of new EPFO subscribers for each year.
#         Query: Get the number of new subscribers aged 18-21 for each month.
#         Query: Find the month with the highest number of establishments remitting their first ECR.
#         Query: Show the total new subscribers by age group for the year 2019-20.

#         40. asi_imp_principal_characteristics_by_rural_urban_sector: This table provides annual data on principal industrial characteristics, such as the number of factories, split by rural and urban sectors in India.
#         Instructions: Use this table to analyze trends or compare rural and urban industrial characteristics (like number of factories) across different years.
#         Example queries:
#         Query: Show the number of factories in rural and urban areas for each year.
#         Query: List all principal characteristics available for 2012-13.
#         Query: Get the total number of factories in rural areas between 2010-11 and 2012-13.

#         41. asi_imp_principal_characteristics_india_by_mjr_indus_grp: This table provides annual data on the number of factories in India, categorized by major industry groups, as reported by the Enterprise Survey Division of MoSPI.
#         Instructions: Use this table to analyze trends or compare the number of factories across different industries and years. Filter by 'year' for specific periods or by industry columns for sector-specific insights. The 'characteristics' column describes the metric (e.g., 'No. of Factories').
#         Example queries:
#         Query: Show the number of factories in the food products industry for each year.
#         Query: Which year had the highest number of factories in the textiles sector?
#         Query: List the number of factories for all industries in 2011-12.
#         Query: Show the trend of factories in the pharmaceuticals industry over the years.

#         42. asi_industrywise_factories_2022_23: This table provides the number of factories in India for various types of industries for the year 2022-23, along with metadata such as release date, last update, and data source.
#         Instructions: Use this table to find the count of factories by industry type, or to analyze industry-wise distribution of factories for the year 2022-23.
#         Example queries:
#         Query: Show the number of factories for each industry type.
#         Query: Which industry has the highest number of factories?
#         Query: List all industries with more than 10,000 factories.
#         Query: What is the total number of factories across all industries?
#         Query: Show the data source and last updated date for the industry-wise factories data.

#         43. asi_no_of_workers_and_person_engaged: This table provides annual data on the number of workers and total persons engaged in enterprises, along with release and update information and the data source.
#         Instructions: Use this table to retrieve historical statistics on workforce size and total engagement in enterprises for specific years or to analyze trends over time.
#         Example queries:
#         Query: Show the number of workers and total persons engaged for the year 1982-83.
#         Query: List all years with their corresponding number of workers.
#         Query: Find the year with the highest total persons engaged.

#         44. asi_num_of_factories_nva: This table provides annual data on the number of factories and their net value added (in lakhs) as recorded by the Enterprise Survey Division, MoSPI.
#         Instructions: Use this table to analyze trends in the number of factories and their net value added over different years. You can filter by year, aggregate values, or compare data across years.
#         Example queries:
#         Query: Show the net value added and number of factories for each year.
#         Query: Find the year with the highest net value added.
#         Query: Get the total net value added across all years.
#         Query: List all years where the number of factories was below 100,000.

#         45. asi_statewise_number_of_factories_for_2022_23: This table provides the number of factories in each Indian state for the year 2022-23, along with release and update dates and the data source.
#         Instructions: Use this table to retrieve or analyze the count of factories by state for 2022-23, or to filter by release/update dates or data source.
#         Example queries:
#         Query: Show the number of factories in each state.
#         Query: Which state had the highest number of factories in 2022-23?
#         Query: List all states where the number of factories is greater than 30,000.
#         Query: Give me the release date for the factory data.

#         46. asi_top_ten_states_by_number_of_factories: This table shows the number of factories in the top ten Indian states for each year from 2010-11 to 2022-23, along with metadata about data release and source.
#         Instructions: Use this table to analyze trends, compare, or retrieve the number of factories in leading Indian states across different years.
#         Example queries:
#         Query: Which state had the highest number of factories in 2022-23?
#         Query: Show the number of factories in Maharashtra and Gujarat for the years 2010-11 and 2022-23.
#         Query: List the trend of factories in Tamil Nadu from 2010-11 to 2022-23.
#         Query: Which state saw the largest increase in the number of factories between 2010-11 and 2022-23?

#         47. asi_trend_imp_characteristics_technical_coefficients: This table contains annual technical coefficients data, such as Fixed Capital to Output ratios, for different years, including metadata like release and update dates and data source.
#         Instructions: Use this table to retrieve technical coefficients by year, region, or data source, or to analyze trends in industrial technical characteristics over time.
#         Example queries:
#         Query: Show the technical coefficients for all years for Fixed Capital to Output.
#         Query: List all available regions in the table.
#         Query: Get the technical coefficient for the year 2012-13.
#         Query: Find the latest update date for the data.

#         48. asi_trend_of_imp_characteristics_structural_ratios: This table provides annual data on key structural ratios, such as fixed capital per factory, across different regions and years, sourced from the Enterprise Survey Division MoSPI.
#         Instructions: Use this table to analyze trends in structural ratios (e.g., fixed capital per factory) over time or by region. Filter by year, region, or data source as needed.
#         Example queries:
#         Query: Show the fixed capital per factory for each year.
#         Query: List all available regions in the table.
#         Query: Get the structural ratios for the year 2011-12.

#         49. asi_trend_of_imp_principal_characteristics_india: This table contains annual data on key industrial characteristics in India, such as the number of factories, with associated metadata including release and update dates, and data source.
#         Instructions: Use this table to retrieve historical trends and statistics related to principal industrial characteristics (e.g., number of factories) in India by year.
#         Example queries:
#         Query: Show the number of factories in India for each year.
#         Query: Get all available data for the year 1982-83.
#         Query: List the years and values for all records sourced from 'Enterprise Survey Division MoSPI'.

#         50. lpfr_state_age: The 'lpfr_state_age' table provides Labour Force Participation Rate (LFPR) data by state, age group, gender, and area (rural/urban/total) for specified time periods.
#         Instructions: Use this table to analyze LFPR statistics across Indian states, broken down by age group, gender, rural/urban/total populations, and time period. Filter by 'state', 'age_group', 'year', or area/gender columns as needed.
#         Example queries:
#         Query: Show the total LFPR for males and females in Andhra Pradesh for April-June 2025.
#         Query: List rural and urban LFPR for persons aged 15-29 in Assam in 2025.
#         Query: Get LFPR for all states for females in rural areas for 2025.

#         51. cws_industry_distribution_state: This table provides the percentage distribution of workers by industry sector (agriculture, secondary including mining and quarrying, tertiary, and all sectors) across Indian states, segmented by age group, gender, area type, and time period.
#         Instructions: Use this table to analyze the sectoral distribution of workers in different states, filtered by age group, gender, area type, and specific time periods. Useful for comparing employment patterns across sectors and demographics.
#         Example queries:
#         Query: Show the percentage of male workers in the agriculture sector in rural Bihar for April to June 2025.
#         Query: List the sector-wise distribution for all sectors in Assam for rural males aged 15 years and above in Q2 2025.
#         Query: Get the tertiary sector percentage for Andhra Pradesh for rural males aged 15 years and above in the latest available data.

#         52. wpr_state_age: This table contains state-wise Worker Population Ratio (WPR) data by gender, area (rural/urban/total), and age group, along with time period, year, data source, and state information.
#         Instructions: Use this table to analyze or retrieve WPR statistics by state, gender, area, age group, and time period. Filter by columns like 'state', 'year', 'age_group', or area/gender-specific WPR values as needed.
#         Examples queries
#         Query: Show the total WPR for males and females in Assam for the age group 15-29 years in 2025.
#         Query: List all states with rural female WPR above 20 for April-June 2025.
#         Query: Get the urban person WPR for Bihar for the latest released data.

#         53. ur_state_age: This table provides state-wise unemployment rates segmented by age group, gender, and rural/urban areas, along with time period and data source details.
#         Instructions: Use this table to analyze or retrieve unemployment rates by state, age group, gender, area (rural/urban/total), and time period (month, year). Filter by columns such as state, year, age_group, or area-specific rates as needed.
#         Examples queries:
#         Query: Show the total unemployment rate for persons aged 15-29 years in Andhra Pradesh for April to June 2025.
#         Query: List rural female unemployment rates for all states in 2025.
#         Query: Get urban male and female unemployment rates in Bihar for 2025.
#         Query: Find the data source and release date for Assam's unemployment data for April to June 2025.


#         54. none_of_these: for any queries which are unrelated to above files.

#         ## Consider the list above, and respond ONLY with one of the file names from the following list:
#         [asuse_est_annual_emoluments_per_hired_worker, asuse_est_annual_gva_per_establishment, asuse_est_num_establishments_pursuing_mixed_activity, asuse_est_num_workers_by_employment_gender, asuse_est_value_key_characteristics_by_workers, asuse_estimated_annual_gva_per_worker_rupees, asuse_estimated_number_of_workers_by_type_of_workers, asuse_per1000_estb_by_hours_worked_per_day, asuse_per1000_estb_by_months_operated_last_365days, asuse_per1000_estb_registered_under_acts_authorities, asuse_per1000_estb_using_computer_internet_last365_days, asuse_per1000_of_estb_using_internet_by_type_of_its_use,
#         asuse_per1000_proppartn_estb_by_edu_owner_mjr_partner, asuse_per1000_proppartn_estb_by_other_econ_activities, asuse_per1000_proppartn_estb_by_socialgroup_owner, asuse_per_1000_distri_of_establishments_by_nature_of_operation, asuse_per_1000_distri_of_establishments_by_type_of_location, asuse_per_1000_distri_of_establishments_by_type_of_ownership, asuse_per_1000_of_establishments_which_are_npis_and_non_npis, asuse_statewise_est_num_of_estb_pursuing_mixed_activity, asuse_statewise_est_num_of_estb_serving_as_franchisee_outlet, asuse_statewise_est_num_of_worker_by_employment_and_gender,
#         asuse_statewise_estimated_annual_emoluments_per_hired_worker, asuse_statewise_estimated_annual_gva_per_establishment_rupees, asuse_statewise_estimated_annual_gva_per_worker_rupees, asuse_statewise_estimated_number_of_workers_by_type_of_workers, asuse_statewise_per1000_distri_of_estb_by_nature_of_operation, asuse_statewise_per1000_distri_of_estb_by_type_of_location, asuse_statewise_per1000_distri_of_estb_by_type_of_ownership, asuse_statewise_per1000_estb_by_hours_worked_per_day, asuse_statewise_per1000_estb_by_month_num_operated_last365_day, asuse_statewise_per1000_estb_maintain_post_bank_saving_acc,
#         asuse_statewise_per1000_estb_registered_diff_acts_authorities, asuse_statewise_per1000_estb_use_computer_internet_last365_day, asuse_statewise_per1000_proppart_estb_by_social_grp_mjr_prtner, annual_survey_of_industries , periodic_labour_force_survey, asi_state_principal_characteristics, epfo_india_mth_payroll, asi_imp_principal_characteristics_by_rural_urban_sector, asi_imp_principal_characteristics_india_by_mjr_indus_grp, asi_industrywise_factories_2022_23, asi_no_of_workers_and_person_engaged, asi_num_of_factories_nva, asi_statewise_number_of_factories_for_2022_23,
#         asi_top_ten_states_by_number_of_factories, asi_trend_imp_characteristics_technical_coefficients, asi_trend_of_imp_characteristics_structural_ratios, asi_trend_of_imp_principal_characteristics_india, lpfr_state_age, cws_industry_distribution_state, wpr_state_age, ur_state_age, none_of_these]

#         Do not include any reasoning traces or other text apart from the file name selected from the above list.
#             """)
#     selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
#     return selected_file.strip(), i_tokens, o_tokens

def file_selector_social_migration_and_households(query):
    system_instruction=dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        1. mis_access_to_improved_source_of_drinking_water: This table presents state-wise and all-India data on the percentage of persons with access to piped water into dwelling/yardplot or improved sources of drinking water, categorized by sector (urban, rural, all). Data include examples like Goa (urban, 100% improved source), Punjab (all, 70.4% piped water), and West Bengal (urban, 40.6% piped water). Information is collected annually and sourced from the National Sample Survey Office, MoSPI.
        Sample queries:
        a. What percentage of rural households in West Bengal have access to piped water into dwelling or yardplot?
        b. Compare the percentage of persons with access to improved sources of drinking water in urban Maharashtra and urban Karnataka.


        2. mis_access_to_mass_media_and_broadband: This table presents state- and all-India–level annual data on the percentage of households reporting access to broadband within premises and to mass media (including Internet, newspaper, magazine, radio, television). Data is classified by region (urban/rural/all) and type (e.g., Broadband, Mass media). Examples include broadband access in Bihar (Urban) at 43%, mass media access in Gujarat (Rural) at 74.4%, and broadband access in Kerala (All) at 76.2%.
        Sample queries:
        a. What percentage of rural households in Uttar Pradesh have access to mass media according to the latest available data?
        b. How does the percentage of households with broadband access in urban Manipur compare to all-India urban levels?


        3. mis_availability_of_basic_transport_and_public_facility: This table provides state-wise and all-India data on the percentage distribution of households with access to basic transport and public facilities, such as living in pucca structures (e.g., 79.5% rural All India, 98.8% Uttarakhand urban) and proximity to all-weather roads or open public spaces (e.g., 93.93% rural Maharashtra, 44.9% urban Goa). Data resolution is by state/UT, rural/urban/all sectors, and the annual frequency reflects survey results reported by MoSPI.
        Sample queries:
        a. What percentage of rural households in Assam have access to all-weather roads within 2 km of their residence?
        b. Which states have more than 95% urban households living in pucca structures?


        4. mis_different_source_of_finance: This table presents the state-wise and all-India percentage distribution of funding sources used by households for new house or flat purchases, with urban, rural, and total resolutions. Data covers financing categories like 'Bank', 'Own finance', 'Private finance', and 'Any other source' (e.g., Kerala-Urban, Bank: 53.4%; Assam-All, Own finance: 64.4%). The dataset is provided by the National Sample Survey Office and appears to be annual, with data as recent as 2025.
        Sample queries:
        a. What percentage of households in Assam used their own finance to buy a new house or flat?
        b. How does the reliance on bank finance for new house purchases differ between urban and rural Bihar?


        5. mis_exclusive_access_to_improved_latrine: This table presents state-wise and urban/rural-wise annual percentages of persons in India with access to improved latrines and exclusive access to improved latrines, as reported by the National Sample Survey Office (MoSPI). Covered states include Manipur (urban 100%), Kerala (rural 99.8%), Telangana (all 95.3%), Odisha (urban 87.3%), and Andaman & Nicobar Islands (all 92.9%). Data is available for all Indian states and union territories for 2023.
        Sample queries:
        a. What percentage of persons in Karnataka have exclusive access to improved latrine facilities?
        b. Compare access to improved latrine in rural vs urban areas of Andhra Pradesh for 2023.


        6. mis_household_assets: This table contains state-wise and area-wise (urban, rural, all) data on the percentage of Indian households that have purchased newly constructed residential houses/flats after 31/03/2014 for the first time and owned them as of the survey date. Data is sourced from the National Sample Survey Office (MoSPI) and reported annually. Examples include Telangana (All, 99.3%), Odisha (Urban, 58.1%), and Andaman & Nicobar Islands (All, 9.7%).
        Sample queries:
        a. What percentage of urban households in Sikkim purchased a new residential flat after March 2014 and still own it?
        b. Which Indian state had the highest percentage of rural households purchasing their first new house after 31 March 2014?


        7. mis_improved_latrine_and_hand_wash_facility_in_households: This table provides state-wise, all-India, and urban/rural data on the percentage distribution of persons by availability and type of hand washing facility within their premises, based on the National Sample Survey Office (NSSO) data. Information is annual and includes categories such as 'Wash Hands with Water Only', 'Wash Hands with Water and Soapdetergent', 'no Hand Washing Facility', and combinations thereof. Examples include Meghalaya (Rural, 21.6% no facility), Goa (Urban, 99.9% with water and soap), and All India figures.
        Sample queries:
        a. What percentage of rural households in Meghalaya have no hand washing facility within the premises?
        b. How many urban persons in Goa have access to hand washing with water and soapdetergent according to the latest NSSO data?


        8. mis_improved_source_of_drinking_water_within_household: This table presents state-wise (and all-India) annual data on the percentage of persons with access to improved sources of drinking water, including sufficiency and exclusivity, as reported by the National Sample Survey Office (MoSPI). Categories by rural, urban, and 'all' are included; examples are West Bengal (rural: 35.1%), Maharashtra (all: 77.0%), and All India (all: 52.2%) for different types of drinking water access as of July 2025.
        Sample queries:
        a. What percentage of rural households in Uttarakhand had exclusive access to improved drinking water sources sufficiently available throughout the year in 2025?
        b. Compare the percentage of urban households with improved drinking water access in Gujarat and Goa according to the latest available data.


        9. mis_income_change_due_to_migration: This table contains state-wise and all-India data on the percentage distribution of persons reporting as earners in their last usual place of residence, categorized by change in income due to migration (increased, decreased, or unchanged). Data are provided separately for rural, urban, and all regions. Notable entries include 'Tamil Nadu - Rural - Increased', 'Jharkhand - Urban - Decreased', and 'All India - Rural - Increased'. Source: NSSO, MoSPI. Data is annual as of 2025.
        Sample queries:
        a. Which states reported the highest increase in earners' income due to migration in 2025?
        b. Show the percentage distribution for persons with decreased income due to migration in urban Jharkhand.


        10. mis_main_reason_for_leaving_last_usual_place_of_residence: This table presents state-wise and gender-wise data on the percentage distribution of persons leaving their last usual place of residence for varied reasons (e.g., post-retirement, acquisition of own house, social-political problems, employment search, marriage) across rural, urban, and all populations in Indian states like Gujarat, Maharashtra, Assam, and West Bengal. The data originates from the National Sample Survey Office, MoSPI, and is reported annually.
        Sample queries:
        a. What percentage of rural males in Gujarat left their last usual place of residence in search of employment?
        b. Which states reported high percentages for 'acquisition of own house/flat' as the main reason for migration among urban females?


        11. mis_main_reason_for_migration: This table presents state-wise, sector-wise (urban/rural/all), and gender-wise annual data on the percentage of persons in India willing to move out from their current residence and the main reasons for such willingness, compiled by the National Sample Survey Office (MoSPI). Reasons cited include employment, marriage, studies, health care, social issues, and more. Sample entries: Gujarat–Rural–Female–Marriage; Rajasthan–All–Male–In search of employment; All India–Urban–Person–Health care.
        Sample queries:
        a. What percentage of urban males in Rajasthan are willing to move out in search of employment?
        b. Which states reported the highest percentage of people willing to leave due to marriage?


        12. mis_possession_of_air_conditioner_and_air_cooler: This table provides state-wise, rural/urban/all, and national-level data on household possession of air conditioners and air coolers in India, sourced from the National Sample Survey Office (MoSPI). It includes metrics like percentage of households owning air conditioners or coolers (e.g., 91% in urban Jharkhand, 0% in urban Sikkim), and average number owned by reporting households, collected annually with data as recent as July 2025 and February/March 2023.
        Sample queries:
        a. What percentage of urban households in Gujarat report having an air cooler?
        b. Which state has the highest average number of air conditioners per reporting household according to the latest data?


        13. mis_usage_of_mobile_phone: This table provides the percentage of persons aged 15 or 18 years and above who used a mobile telephone with an active SIM card at least once during the three months preceding the survey. Data is available at national, state/UT, and all-India levels, disaggregated by gender (Male, Female, Person) and sector (Urban, Rural, All). Examples include Karnataka, West Bengal, Assam, and Chandigarh. The data reference period is annual (as of March 7, 2023).
        Sample queries:
        a. What was the percentage of rural females using mobile phones with active SIM cards in Bihar according to the latest survey?
        b. How does mobile usage among persons aged 18 years and above in urban areas of Maharashtra compare to the all-India average?


        14. mis_usual_place_of_residence_different_from_current_place: This table presents state-wise and area-wise (rural, urban, and all-areas) estimates for the 'Percentage of Persons Whose Current Place of Residence is Different From the Last Usual Place of Residence' across India. Data is available at the state/UT and area levels for a recent annual reference period. Examples include Chandigarh (53.1%), Andaman and Nicobar Islands (47.0%), Andhra Pradesh (35.0%), and area breakdowns such as Rajasthan (Urban, 27.0%) and Bihar (Rural, 18.5%).
        Sample queries:
        a. What is the percentage of people who have changed their usual place of residence in Jharkhand's urban areas?
        b. Which state has the highest percentage of residents whose current residence differs from their last usual residence?

        15. aadhaar_demographic_monthly_data: This table contains monthly Aadhaar demographic data by date, state, district, and pincode, including counts for age groups 5-17 and 17+.
        Instructions: Use this table to analyze Aadhaar demographic trends by location, date, and age group. Filter by state, district, pincode, or date as needed to get specific insights.
        Example queries:
        Query: Show total Aadhaar registrations for age group 17+ in Rajasthan for January 2025.
        Query: List all districts in Chhattisgarh with their Aadhaar registrations for age group 5-17 on 2025-01-03.
        Query: Get the total number of Aadhaar registrations for each state on 2025-01-03.


        16. aadhaar_biometric_monthly_data: This table contains monthly Aadhaar biometric update counts, broken down by age group (5-17 and 17+), for each pincode, district, and state.
        Instructions: Use this table to analyze or retrieve Aadhaar biometric update statistics by date, location (state, district, pincode), and age group.
        Example queries:
        Query: Show the total number of biometric updates for people aged 17 and above in Maharashtra in January 2025.
        Query: List the districts in Tamil Nadu with more than 100 biometric updates for ages 5-17 on 2025-01-03.
        Query: Get the total biometric updates (all ages) per state for the latest date available.


        17. cghs_approved_hospital_data: This table contains information about CGHS approved diagnostic centres and hospitals, including their names, addresses, and the cities they are located in.
        Instructions: Use this table to find details about CGHS approved hospitals or diagnostic centres, such as their names, addresses, or to filter them by city.
        Example queries:
        Query: List all CGHS approved hospitals in Hyderabad.
        Query: Show the addresses of diagnostic centres in Mumbai.
        Query: Get the names and cities of all CGHS approved diagnostic centres.

        18. labour_india_sector_industry_occupation_wages: This table contains wage data for various occupations across different industries and sectors in India, with details on base year, year, and reporting period.
        Instructions: Use this table to analyze or retrieve wage information by sector, industry, occupation, year, or reporting period.
        Example queries:
        Query: Show the absolute wages for all occupations in the Sugar industry for 2023.
        Query: Get the wages of Fitters in the Manufacturing Sector for the year 2023.
        Query: List all available occupations and their wages as of 1st January 2023.

        19. labour_india_rural_wages: This table contains rural wage data in India, categorized by year, month, state, occupation, and item, with separate wage values for men and women.
        Instructions: Use this table to analyze rural wage trends in India by gender, state, occupation, or time period.
        Example queries:
        Query: Show the average wage for men and women in agricultural occupations in 2023.
        Query: List the states with the highest average rural wages for women in May 2022.
        Query: Get the monthly wage trend for men in Karnataka for the occupation 'Construction Worker' in 2021.

        20. airport_sewa_services_data: This table contains information about various services available at airports, including service category, title, description, contact details, and the last update date.
        Instructions: Use this table to find details about airport services such as transportation, parking, and other amenities. Filter by airport name, service category, or specific service titles to get relevant contact information and descriptions.
        Example queries:
        Query: Show all car rental services available at Chennai airport.
        Query: List all parking and transportation services at Mumbai airport.
        Query: Get the contact phone numbers for all services at Chennai airport.
        Query: Find all services updated after October 1, 2024.

        21. traffic_india_mth_air_passengers: Monthly air passenger traffic data in India, including year, month, passenger numbers, and last update date.
        Instructions: Use this table to analyze trends in air passenger numbers in India by month and year. Filter or aggregate by year, month, or passenger counts as needed.
        Example Queries :
        Query: Show total air passengers for each month in 2024.
        Query: What is the total number of air passengers in India for 2024?
        Query: List the months with more than 50,000 passengers in 2024.

        22. sp_india_daily_state → This table provides daily statistics for Indian states on Ayushman card creation, hospital admissions, and empanelled hospitals under the Ayushman Bharat scheme.
        Instructions: Use this table to analyze or retrieve state-wise daily data on Ayushman card creation, hospital admissions, and the number of empanelled hospitals. Filter by 'state_name' or 'updated_on' to get specific records.
        Example Queries:
        Query: Show the total Ayushman cards created in Uttar Pradesh as of 2025-10-01.
        Query: List all states with more than 5,000,000 hospital admissions as of 2025-10-01.
        Query: Get the number of empanelled hospitals in Bihar on 2025-10-01.

        23. demography_india_yr_popsexgrowth: This table provides annual demographic statistics for India, including rural, urban, and total population counts by gender, sex ratios, and average annual growth rates from 1951 onwards.
        Instructions: Use this table to analyze population trends, gender distribution, sex ratios, and growth rates in rural, urban, or total populations of India by year.
        Example queries:
        Query: Show the total population and sex ratio for each census year.
        Query: What was the rural female population and sex ratio in 1971?
        Query: List the average annual growth rate of the urban population for all available years.
        Query: Find the year with the highest total sex ratio.

        24. demography_india_state_yr_literacy: This table contains literacy rates for different age groups across Indian states and union territories, broken down by gender and by rural/urban areas.
        Instructions: Use this table to analyze literacy rates by state/UT, age group, gender, and area type (rural/urban/all). Select relevant columns based on the demographic and geographic breakdown you need.
        Example queries:
        Query: Show the overall literacy rate for males and females aged 15-24 in Assam.
        Query: List the rural and urban literacy rates for persons aged 15-24 in Andhra Pradesh.
        Query: Get the literacy rates for all persons in Arunachal Pradesh for the 15-24 age group.

        25. hces_state_yr_assets : This table provides state-wise and year-wise data on household ownership of telephones/mobiles and computers in rural and urban areas across India.
        Instructions: Use this table to analyze the penetration of telephones/mobiles and computers in rural, urban, and overall households by state and year. Filter by 'state_ut' and 'year' for specific regions and periods.
        Example queries:
        User Query: Show the percentage of rural households with computers in Assam for 2022-23.
        User Query: List all states with more than 90% mobile/telephone ownership in rural areas for 2022-23.
        User Query: Get the urban computer ownership percentage for Arunachal Pradesh in 2022-23.
        User Query: Show all columns for Andhra Pradesh for the year 2022-23.
        
        26. tlm_state_yr_transport_access : This table provides annual data on transport access percentages for each Indian state/UT, including urban low and high capacity transport access and rural all-weather road access.
        Instructions: Use this table to retrieve or analyze state-wise or year-wise statistics on urban and rural transport access percentages.
        Example queries:
        User Query: Show the urban low capacity transport access percentage for all states in 2022-23.
        User Query: Which states had rural all-weather road access above 90% in 2022-23?
        User Query: List the urban high capacity transport access percentage for Assam.
        
        27. env_state_yr_river_water_quality : This table contains annual river water quality statistics for Indian states and union territories, including temperature, dissolved oxygen, pH, conductivity, BOD, nitrate, and coliform levels.
        Instructions: Use this table to analyze or retrieve yearly river water quality parameters by state/UT, such as temperature ranges, dissolved oxygen, pH, conductivity, BOD, nitrate, and coliform counts. Filter by 'state_ut' and 'year' as needed.
        Example queries:
        User Query: Show the minimum and maximum dissolved oxygen levels for Assam in 2022.
        User Query: List all states with their maximum BOD values for the year 2022.
        User Query: Get the pH range for Andhra Pradesh for the latest available year.
        User Query: Find states where the maximum nitrate level exceeded 5 mg/l in 2022. 
        
        
        28. none_of_these: for any queries which are unrelated to above tables.

        ## Consider the list above, and respond ONLY with one of the file names from the following list:
        [mis_access_to_improved_source_of_drinking_water, mis_access_to_mass_media_and_broadband, mis_availability_of_basic_transport_and_public_facility, mis_different_source_of_finance,
        mis_exclusive_access_to_improved_latrine, mis_household_assets, mis_improved_latrine_and_hand_wash_facility_in_households, mis_improved_source_of_drinking_water_within_household,
        mis_income_change_due_to_migration, mis_main_reason_for_leaving_last_usual_place_of_residence, mis_main_reason_for_migration, mis_possession_of_air_conditioner_and_air_cooler,
        mis_usage_of_mobile_phone, mis_usual_place_of_residence_different_from_current_place,aadhaar_demographic_monthly_data, aadhaar_biometric_monthly_data, cghs_approved_hospital_data,
        labour_india_sector_industry_occupation_wages, labour_india_rural_wages, airport_sewa_services_data, traffic_india_mth_air_passengers, sp_india_daily_state, demography_india_yr_popsexgrowth, demography_india_state_yr_literacy,
        hces_state_yr_assets, tlm_state_yr_transport_access, env_state_yr_river_water_quality, none_of_these]
        do not include any reasoning traces or other text apart from the file name selected from the above list.
            """)
    selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_CPI(query):
    system_instruction=dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        1. cpi_state_mth_grp_view: This view provides monthly Consumer Price Index (CPI) data by aggregate group or category level. It is the DEFAULT view for state level categories.
        Instructions: Use this view to analyze or retrieve state-wise, sector-wise, and group-wise monthly CPI and inflation rates for different years and months. It is suitable for queries requiring aggregate CPI data at the group level (not sub-group).
        Sample queries:
        a. Show the inflation index for Karnataka (Rural) for Housing in May 2021.
        b. Get the inflation rate for Sikkim in July 2022 for the Food and Beverages group.


        2. cpi_state_mth_subgrp_view: This table/view contains monthly Consumer Price Index (CPI) data by sub-group, for different years and months in India.
        Instructions: Use this table/view to retrieve CPI data at the sub-group level for specific states, sectors (Urban/Rural), months, and years.
        Sample queries:
        a. Show the inflation index for 'Egg' in Bihar (Urban sector) for May 2021.
        b. List all inflation indices for 'Spices' in Jharkhand (Rural) for fiscal year 2020-21.


        3. cpi_india_mth_grp_view: Monthly Consumer Price Index (CPI) data for India at the group level, filtered for 'All India'. This is the DEFAULT view for India level inflation queries.
        Instructions: Use this view to analyze CPI trends, inflation rates, and index values by year, month, sector, and group for the whole of India. Useful for time series analysis, inflation monitoring, and economic research.
        Sample queries:
        a. Show the latest CPI inflation rate for the 'General' group in All India.
        b. Get the CPI index and inflation rate for 'Housing' in Urban sector for April 2024.
        c. Inflation trends in the country in fiscal year 2022-23.


        4. cpi_india_mth_subgrp_view: This view provides monthly Consumer Price Index (CPI) data for India at the sub-group level, filtered for 'All India' and excluding aggregate sub-groups, with details on inflation index and rate by sector, group, and sub-group.
        Instructions: Use this view when India-level queries for named sub-groups (such as vegetables) or "all sub-groups" are raised.
        Sample queries:
        a. What was the inflation rate for 'Fruits' in November 2021 for the Urban sector?
        b. Show the inflation index for 'Education' in November 2024 for All India Urban.
        c. List all sub-groups under 'Food and Beverages' for September 2017 (Combined sector).


        5. consumer_price_index_cpi_for_agricultural_and_rural_labourers: This table covers data for year 2024. it should be used only when "agriculture labour" or "rural labour" is mentioned. DO NOT use this file unless "labour" is specifically mentioned.
        a. What was the minimum wage for agricultural workers in Gujarat in June 2024?
        b. Show the all-India minimum wage for rural labour in Food category for July 2024.


        6. city_wise_housing_price_indices: This table presents city-wise housing price index data from the National Housing Bank for major cities across India, such as Mumbai, Kolkata, Faridabad, Chennai, Bengaluru, Kochi, Ahmedabad, and Bhopal. The information is available at a quarterly frequency for various years (2013 to 2025), including details like city name, price index value (e.g., 110.00 for Mumbai in 2020 Q1), and date of data release. It should be used only when "housing prices" are mentioned.
        Sample queries:
        a. What was the housing price index for Bengaluru in Q1 of 2020?
        b. Show the quarterly price index trend for Kochi from 2016 to 2021.


        7. whole_sale_price_index_wpi_financial_year_wise: this table covers data for year 2012 to 2023. it should be used only when wpi or wholesale prices are mentioned in financial year or fy format.It should be used only when WPI or wholesale prices are mentioned using a financial year format (e.g., "2022-2023", "2023-24", or "FY 2023").
        Sample queries:
        a. What was the wholesale price index for cotton cloth in 2023-24?
        b. What was the wholesale price index across all commodities in 2012-13?


        8. cpi_worker_data: this table covers data for year 2011 to 2023. it should be used only when WORKERS are mentioned. It contains data about Industrial Workers, Rural Labour, Urban Labour, and Agricultural Labour. DO NOT select this table if query is about cpi index for food/non-food workers.
        Sample queries:
        a. What was the CPI-NS value for Rural areas in 2022?
        b. Show the trend of CPI-AL (Agricultural Labourers) index values for the last five years.


        9. whole_sale_price_index_wpi_calendar_wise: this table covers data for year 2013 to 2023.It should be used only when WPI or wholesale prices are mentioned using a calendar year format (e.g., "2023", "in 2022", or just a single year number). The records cover a wide range of products such as flourescent tubes, walnut, fungicides, cotton cloth, wine, black pepper, okra, railway brake gear, and ragi. Each entry includes product codes, weights, base years, index values, and relevant reference periods.
        Sample queries:
        a. What was the wholesale price index for cotton cloth in 2024?
        b. What was the wholesale price index across all commodities in 2013?


        10. cpi_food_worker_data: This table provides annual Consumer Price Index for Industrial Workers (CPI-IW) data for India, categorized by item groups such as 'Food' and 'Non Food'. The dataset includes both 'Average of Months' and 'Last Month of' indices from 2011 to 2020, with figures like 293.00 (Food, 2015) and 317.00 (Non Food, 2019). Do not use this table for queries related to CPI for Agricultural and Rural Labourers/workers.
        Sample queries:
        a. What was the annual CPI-IW value for the 'Food' category in 2018?
        b. How did the 'Non Food' CPI-IW index change from 2011 to 2019 at the national level?

        11. cpi_iw_point_to_point_inflation: This table contains monthly point-to-point inflation rates for the Consumer Price Index for Industrial Workers (CPI-IW), including base year, year, month, inflation value, and data source.
        Instructions: Use this table to retrieve CPI-IW inflation rates for specific months, years, or base years, or to analyze inflation trends over time.
        Example queries:
        Query: Show the CPI-IW inflation rates for the year 2020.
        Query: Get the inflation rate for September 2020.
        Query: List all available base years in the CPI-IW inflation data.
        Query: Find the average CPI-IW inflation for 2020.

        12. cpi_iw_centre_index: This table contains Consumer Price Index for Industrial Workers (CPI-IW) data by centre, state, year, and month, with index values based on a specified base year.
        Instructions: Use this table to retrieve CPI-IW index values for specific centres, states, years, and months, or to analyze inflation trends over time by location.
        Example queries:
        Query: Show the CPI-IW index for Guntur in January 2024.
        Query: List all CPI-IW index values for Andhra Pradesh in 2023.
        Query: Get the yearly CPI-IW index trend for Guntur from 2021 to 2024.

        13. cpi_iw_retail_price_index: This table contains the Consumer Price Index (CPI) for Industrial Workers (IW) retail price index data, including item-wise indices by year, month, group, and sub-group, with data sourced from the Ministry of Labour & Employment.
        Instructions: Use this table to analyze or retrieve CPI-IW retail price index values for specific items, groups, sub-groups, years, or months. You can filter by item, group, sub_group, year, month, or base_year to get relevant index values.
        Example queries:
        Query: Show the CPI index for 'Scents and perfumes' in December 2021.
        Query: List all items under the 'Personal Care & Effects' sub-group for 2021.
        Query: Get the CPI index values for all items in the 'Miscellaneous' group for December 2021.
        Query: Find the average index for 'Personal Care & Effects' in 2021.

        14. hces_india_yr_sector: This table provides average Monthly Per Capita Expenditure (MPCE) data for rural and urban areas across Indian states and union territories, including both original and imputed values.
        Instructions: Use this table to analyze or compare average MPCE values (both original and imputed) for rural and urban populations by state or union territory.
        Example queries:
        Query: Show the average rural and urban MPCE for all states.
        Query: List states where the imputed urban MPCE is greater than 9000.
        Query: Find the difference between rural and urban average MPCE for each state.
        Query: Get the imputed rural MPCE for Assam.

        15. none_of_these: for any queries which are unrelated to inflation. for example, queries regarding gdp, iip, msme would fall under the "none" category. queries regarding the general state of the economy, government policies, and upcoming challenges also fall under the none_of_these category.

        ## Consider the list above, and respond ONLY with one of the file names from the following list:
        [cpi_state_mth_grp_view, cpi_state_mth_subgrp_view, cpi_india_mth_grp_view, cpi_india_mth_subgrp_view, consumer_price_index_cpi_for_agricultural_and_rural_labourers, city_wise_housing_price_indices, whole_sale_price_index_wpi_financial_year_wise, cpi_worker_data, whole_sale_price_index_wpi_calendar_wise, cpi_food_worker_data, cpi_iw_point_to_point_inflation, cpi_iw_centre_index, cpi_iw_retail_price_index, hces_india_yr_sector, none_of_these]
        do not include any reasoning traces or other text apart from the file name selected from the above list.
            """)
    selected_file, i_tokens, o_tokens = llm_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_GST(query):
    system_instruction=dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        1. gross_and_net_tax_collection: This dataset provides a detailed breakdown of gross and net tax collections in India, including Goods and Services Tax (GST) revenue. It covers various categories such as Domestic Refunds,  domestic revenue, import revenue, and total GST revenue, Net Revenue Domestic ,Net IGST Revenue ,Export IGST Refunds , Net Revenue ,Export GST Refunds through ICEGATE,further segmented into sub-categories like Central GST (CGST), State GST (SGST), Integrated GST (IGST), and CESS. The data is available on a monthly and yearly basis, sourced from the Reserve Bank of India (RBI). This dataset is crucial for understanding the trends in tax revenue, the impact of GST on different sectors, and the overall fiscal health of the Indian economy. Keywords: Tax collection, GST, CGST, SGST, IGST, CESS, Revenue, India, Monthly, Yearly, Domestic, Imports, Refunds.Do not choose this if state is given in the querty.
        Query: "What was the total gross GST revenue collected in April 2024?",
        Query: "What was the monthly value of CGST collected from domestic sources in April 2023?",
        Query: "What was the total value of refunds related to IGST in April 2024?",
        Query: "What was the gross import revenue collected in April 2023?",
        Query: "What was the net domestic revenue collected in April 2024?"


        2. gstr_one : This dataset provides a comprehensive overview of Goods and Services Tax (GST) return filing behavior across Indian states and union territories. It captures monthly data on the number of taxpayers eligible to file GST returns (GSTR-1), the number who filed by the due date, the number who filed after the due date, the total number of returns filed, and the filing percentage. The data is organized by fiscal year, month, and state, offering a time-series perspective on GST compliance. This dataset is crucial for understanding GST compliance rates, identifying states with high or low compliance, and analyzing trends in filing behavior over time. The source of the data is also provided, along with the dates when the data was released and updated. Keywords: GST, Goods and Services Tax, tax returns, GSTR-1, filing compliance, India, states, fiscal year, monthly data, taxpayers.Also if the query contains "outward supplies", "sales invoices", or "GSTR-1", select gstr_one.
        Query: "What was the total number of GST returns filed in Maharashtra during April 2021-22?",
        Query: "What percentage of eligible taxpayers in Tamil Nadu filed their GST returns by the due date in August 2021-22?",
        Query: "How many taxpayers were eligible to file GST returns in Gujarat during the month of April in the fiscal year 2021-22?",
        Query: "What was the total number of GST returns filed in the state of Kerala during August of the fiscal year 2021-22?",
        Query: "What was the GST filing percentage for the state of Punjab as of June 30th, 2025, for the month of August in the fiscal year 2021-22?"


        3. gstr_three_b : This dataset, provides a comprehensive overview of Goods and Services Tax (GST) return filing compliance across Indian states and union territories. It includes monthly data on the number of taxpayers eligible to file GST returns (GSTR-3B), the number who filed by the due date, the number who filed after the due date, the total number of returns filed, and the filing percentage. The data is categorized by fiscal year, month, and state, offering a granular view of GST compliance trends. This information is sourced from the Reserve Bank of India (RBI) and is updated periodically. The dataset is valuable for understanding state-wise variations in tax compliance, identifying potential areas of tax evasion, and assessing the overall effectiveness of GST implementation. Keywords: GST, tax compliance, India, state-wise data, return filing, GSTR-3B, fiscal year, monthly data, taxpayers, RBI. Also If the query contains "summary return", "tax payment", or "GSTR-3B", select gstr_three_b and if the query only mentions 'GST returns' without specifying, default to gstr_three_b.
        Query: "What was the total number of GST returns filed in Maharashtra during April of the fiscal year 2025-26?",
        Query: "What percentage of eligible taxpayers in Tamil Nadu filed their GST returns by the due date in May of the fiscal year 2025-26?",
        Query: "How many taxpayers were eligible to file GST returns in Gujarat during May of the fiscal year 2025-26?",
        Query: "How many taxpayers in West Bengal filed their GST returns after the due date in April of the fiscal year 2025-26?",
        Query: "What was the total number of GST returns filed in Andhra Pradesh during May of the fiscal year 2025-26?"


        4. gst_settlement_of_igst_to_states : This dataset provides a detailed breakdown of Goods and Services Tax (GST) settlements from the central government to individual states and union territories in India. Specifically, it captures the monthly Integrated GST (IGST) settlements, which include both 'regular' and 'adhoc' components. IGST is levied on inter-state supply of goods and services, and a portion of this revenue is settled with the state where the goods or services are consumed. The data is organized by fiscal year, month, and state, providing a time-series view of these settlements. This dataset is crucial for understanding the fiscal dynamics between the central government and the states, monitoring state revenue streams, and assessing the impact of GST on state finances. Keywords: GST, IGST, settlement, states, fiscal year, month, revenue, India.
        Query: "What was the total GST settlement amount for Maharashtra in the fiscal year 2024-25?",
        Query: "What was the regular GST settlement amount for Tamil Nadu in January of 2026-27?",
        Query: "What was the total GST settlement amount for the state of Gujarat in the month of June in the fiscal year 2023-24?",
        Query: "What was the adhoc GST settlement amount for the state of Kerala in the fiscal year 2022-23?",
        Query: "What was the total GST settlement amount for Punjab in December of the fiscal year 2024-2025?"


        5. gst_statewise_tax_collection_data : This dataset provides a comprehensive record of Goods and Services Tax (GST) collections for each state and union territory in India, broken down by fiscal year and month. It includes the collection amounts for Central GST (CGST), State GST (SGST), Integrated GST (IGST), and CESS. The data is sourced from the Reserve Bank of India (RBI) and includes the dates when the data was released and last updated. This dataset is crucial for understanding the revenue generation of individual states and the overall impact of GST on the Indian economy. It allows for analysis of tax collection trends over time, comparison of performance across different states, and assessment of the contribution of different components of GST to the total revenue. Keywords: GST, tax collection, state revenue, CGST, SGST, IGST, CESS, fiscal year, monthly data, RBI, India.
        Query: "What was the total GST collected in Maharashtra during the fiscal year 2022-23?"
        Query: "What was the SGST collected in Tamil Nadu in the month of December 2021?"
        Query: "What was the CESS collected in Karnataka during the fiscal year 2020-21?"
        Query: "What was the IGST collected in Delhi during the month of June 2022?"
        Query: "What was the total GST collected in India (sum of all states) during the fiscal year 2023-24?"
        Query: "Top 5 GST contributing states in India?"


        6. gst_statewise_tax_collection_refund_data : This dataset provides a detailed, state-wise breakdown of Goods and Services Tax (GST) refunds in India. It includes the amount refunded under various GST components: Central GST (CGST), State GST (SGST), Integrated GST (IGST), and CESS. The data is organized by fiscal year and month, offering a time-series perspective on GST refunds across different states. The data source is specified, typically the Reserve Bank of India (RBI). This dataset is crucial for understanding the flow of GST refunds, assessing the efficiency of the GST system, and analyzing the financial health of individual states. Keywords: GST, Goods and Services Tax, refunds, state-wise, CGST, SGST, IGST, CESS, fiscal year, monthly data, RBI, India, tax refunds, indirect tax.
        Query: "What was the total GST refund amount for Maharashtra in the fiscal year 2022-23?",
        Query: "What was the amount of IGST refund for Tamil Nadu in January 2023?",
        Query: "What was the CESS refund amount for Karnataka in the fiscal year 2021-22?",
        Query: "What was the total refund amount for all states in December 2022?",
        Query: "What was the SGST refund amount for Gujarat in the month of June in the fiscal year 2022-23?"


        7. gst_registrations : This dataset provides a snapshot of Goods and Services Tax (GST) registrations across India, disaggregated by state and taxpayer type. It includes counts of normal taxpayers, composition taxpayers (small businesses with simplified compliance), input service distributors, casual taxpayers, tax collectors at source (TCS), tax deductors at source (TDS), non-resident taxpayers, Online Information Database Access and Retrieval (OIDAR) service providers, and Unique Identity Number (UIN) holders (e.g., embassies). The 'total' column represents the sum of all registration types within each state. This data is crucial for understanding the distribution of businesses registered under GST, assessing the adoption of different GST schemes, and analyzing regional economic activity. Keywords: GST, registrations, taxpayers, state, India, composition scheme, TCS, TDS, OIDAR, UIN.
        Query: "What is the total number of GST registrations in Maharashtra?",
        Query: "How many composition taxpayers are registered in Uttar Pradesh?",
        Query: "What is the number of normal taxpayers registered in Tamil Nadu?",
        Query: "What is the count of tax collectors at source in Gujarat?",
        Query: "How many input service distributors are registered in Karnataka?"


        8. gst_statewise_fiscal_year_collection_view : This table provides annual state-wise data for India on financial metrics (in crore rupees) related to 'RBI Goods and Services', covering multiple years (e.g., 2021-22 to 2025-26). States include Delhi, Assam, Gujarat, Nagaland, Sikkim, and all-India entities like CBIC. Each row details yearly values for different financial categories, the source, and update/publish dates, enabling time-series comparisons across different Indian states.
        “What was the total GST collection of Maharashtra in FY 2022-23?”
        “Which state had the highest GST collection in FY 2021-22?”
        “Show me the GST collection breakdown (CGST, SGST, IGST, Cess) for Tamil Nadu in FY 2020-21.”
        “What is the total GST revenue collected across India in FY 2022-23?”
        “List the top 5 states by GST collection in FY 2019-20.”


        9. gst_statewise_fiscal_year_igst_settlement_view : This table presents annual, state-wise data across Indian states and union territories, detailing figures for variables such as total values, adjustments, and final amounts for 'RBI Goods and Services' (e.g., Gujarat: 27,659; Kerala: 10,939; Lakshadweep: 23). Data spans years like 2020-21 to 2025-26, covers small and large states, and includes all-India territories. Entries feature update dates for each period.
        “How much regular IGST settlement was made to Gujarat in FY 2022-23?”
        “Which state received the highest total IGST settlement in FY 2020-21?”
        “Show me the adhoc IGST settlements for Karnataka in FY 2021-22.”
        “What is the total IGST settlement disbursed across India in FY 2022-23?”
        “Rank the top 3 states by IGST settlement amount in FY 2019-20.”


        10. gst_statewise_fiscal_year_refund_view : This table provides annual, state/UT-wise data on monetary values related to 'RBI Goods and Services' across India. Each row details figures for a financial year, with examples like Gujarat (2023-24), Tamil Nadu (2024-25), and Andaman and Nicobar Islands (2023-24). All-India resolution is included with states and territories as unique entries. Data includes values for different categories over multiple years, last updated in 2025.
        “Give me the SGST refund amount for Karnataka in FY 2022-23.”
        “What is the total refund disbursed across India in FY 2021-22?”
        “Which state received the highest GST refunds in FY 2020-21?”
        “Show me the refund split (CGST, SGST, IGST, Cess) for Maharashtra in FY 2019-20.”
        “List the bottom 5 states by refund amounts in FY 2022-23.”
        
        11. ewb_state_mth_stats : Monthly statistics of e-way bills for each state, including counts and assessable values for within-state, outgoing, and incoming transactions.
        Instructions: Use this table to analyze e-way bill activity by state, year, and month, including supplier counts and assessable values for within-state, outgoing, and incoming movements.
        Example queries:
        User Query: Show the total number of e-way bills generated within Punjab in August 2025.
        User Query: List the assessable value of incoming e-way bills for all states for August 2025.
        User Query: Get the number of outgoing suppliers for Himachal Pradesh in August 2025.
        User Query: Show all columns for Jammu and Kashmir for August 2025.
        


        12. none_of_these: for any queries which are unrelated to GST or taxes. for example, queries regarding gdp, iip, msme would fall under the "none" category. queries regarding the general state of the economy, government policies, and upcoming challenges also fall under the none_of_these category.


        ## Consider the list above, and respond ONLY with one of the file names from the following list:
        [gst_registrations, gst_statewise_tax_collection_refund_data, gst_statewise_tax_collection_data, gst_settlement_of_igst_to_states, gstr_three_b, gstr_one, gross_and_net_tax_collection, gst_statewise_fiscal_year_collection_view, gst_statewise_fiscal_year_igst_settlement_view, gst_statewise_fiscal_year_refund_view, ewb_state_mth_stats, none_of_these]
        Do not include any reasoning traces or other text apart from the file name selected from the above list.
            """)
    selected_file, i_tokens, o_tokens = llm_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_GDP(query):
    system_instruction=dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        # Important rules
        - DO NOT USE annual_estimate_gdp_crore table to answer provisional data related queries.
        - For any queries LONGER THAN 2 years duration, pick ANNUAL or YEARLY tables where available.
        - All questions about "top k states" should go to gdp_state_fy_actuals
        - All questions about "states driving economic growth" should go to gdp_state_fy_actuals

        # Table data

        1. gdp_india_fy_estimates_view : Aggregated national-level GDP metrics by financial year. Captures GDP and GVA (Gross Value Added) growth at constant and current prices. Any India level summarization needs for GDP across years, trending and growth rate can be done from this table.
        Query1: "Compare GDP growth at constant vs current prices for the last 5 years."
        Query2: "What was India's GDP and its growth rate in the most recent year?"
        Query3: "How has the GDP growth rate at constant prices evolved over the last 10 years?"


        2. india_fy_gdp_components_view : Break-up of GDP by economic components (e.g., PFCE, GFCE,GFCF,CIS,VALUABLES, DISCREPANCIES etc) for each FINANCIAL YEAR at INDIA or NATIONAL level. Supports analysis related to components of GDP at year level and share analysis.
        Query: "Give the value of PFCE for the years 2020-21 to 2024-25."
        Query: "How has capital formation (GFCF) changed over the past decade?"
        Query: "Compare the growth of imports and exports of goods and services in the latest year."
        Query: "Show me the trend of GFCE in constant prices over the last 5 years."
        Query: "Which GDP component had the highest growth in 2021-22?"
        Query: "What was the contribution of CIS and VALUABLES to the GDP in 2023-24?"


        3. india_fy_national_income_view : National income accounting components both for Gross and Net Income levels with growth rates at both constant and current prices, financial year-wise.
        Query: "Show me the Net National Income from 2014-15 to 2020-21."
        Query: "Compare the growth rate of Gross National Income over the past 5 years."
        Query: "What was the GNI and NNI in the year 2023-24?"
        Query: "How did the Net National Income at constant prices trend over the last decade?"
        Query: "Give me the latest Gross National Income in current prices."
        Query: "When did GNI show negative growth at constant prices?"


        4. gdp_state_fy_actuals : Use this table when the query is about yearly, financial year, state-level economic performance, specifically when it refers to state domestic product (GSDP) or gross value added (GSVA) or growth rates of different states. Note that this data contains ANNUAL information. Always use this table when specific industrial sectors in states are not mentioned, e.g. "what is the GDP of Maharashtra last year". "Top 3 states by GDP". "States driving India's economic growth." "Population used for GSDP calculation".
        Query: "What was the GSDP of Maharashtra last year?"
        Query: "Top 5 states by GSDP in the most recent year."
        Query: "Show the year-on-year GSDP growth rate for Kerala from 2018-19 to 2023-24."
        Query: "Compare the GSVA at constant prices for Telangana over the past decade."
        Query: "Which states recorded the highest per-capita GSDP in 2024-25?"
        Query: "States driving India’s economic growth in the latest available year."
        Query: "Population used for GSDP calculation"


        5. gdp_state_fy_industry_actuals_view : Use this table when the query is about sectors and industries within states. These components include sectors such as electricity, construction, financial services, transport, etc. Do NOT use this table when no sector is mentioned in the query.
        Query: "What is the contribution of manufacturing to Tamil Nadu's GSDP in 2022-23?"
        Query: "Compare the construction sector output of Gujarat and Maharashtra over the last 5 years."
        Query: "Show the trend of electricity sector growth in West Bengal from 2011-12 to 2024-25."
        Query: "How has the financial services industry performed in Karnataka recently?"
        Query: "What was the value added by agriculture in Bihar during 2020-21?"
        Query: "Give a sector-wise GSDP breakdown for Rajasthan for 2023-24."
        Query: "Compare the trade and repair sector across northeastern states in the most recent year."
        Query: "Subsidies on products for South Indian states"


        6. per_capita_income_product_final_consumption : This table contains per capita estimates of key economic indicators in ₹ (Indian Rupees) or growth rate (%) for various years at India Level, along with the population used for those calculations. The indicators relate to income and consumption at both current and constant prices.these measures are ["Per Capita GDP","Per Capita GNI","Per Capita NNI","Per Capita GNDI","Per Capita PFCE" , "Percentage change over previous year at constant (2011-12) prices"]
        Query: "What is the current per capita income in India?"
        Query: "Give the per capita GDP growth rate for the last five years."
        Query: "What was the per capita private final consumption expenditure in 2014-15?"
        Query: "How has per capita GNI changed from 2011-12 to 2024-25?"
        Query: "Provide the trend of per capita NNI in constant prices over the last decade."
        Query: "Population used for calculating per capita indicators in 2020-21?"
        Query: "Show per capita GNDI values and growth rates since 2015."


        7. top_fifty_macro_economic_indicators_weekly_data: Resolve to this table when the query relates to macroeconomic indicators tracked on a weekly basis for the Indian economy. Focus areas include monetary policy instruments, interest rates (such as repo rate, bank rate, MSF, base rate), yield on government securities and treasury bills, cash reserve ratio (CRR), statutory liquidity ratio (SLR), standing deposit facility (SDF), and forward premia of the US dollar for different durations. Queries about foreign exchange reserves, liquidity conditions, or financial market trends across specific weeks, months, or quarters also belong here. Questions that ask how these economic indicators have changed over time, what the rates were during a particular period, or comparisons between indicators like G-Sec yields and T-bill rates should map to this dataset. Use this table when the intent is to understand the financial health, monetary policy stance, or interest rate environment in India over time.
        Sample queries:
        a. What was the repo rate and reverse repo rate in India on January 27, 2023?
        b. How did the all-India aggregate monetary value change between January 2021 and January 2024?


        8. top_fifty_macro_economic_indicators_quaterly_data:  Resolve to this table when the query includes terms like balance of payments, BoP, overall BoP, net BoP, international investment position, external debt, Indias external debt, gross external debt, or mentions of quarterly external sector data. Use this table if the query asks for the net value of balance of payments, status of India’s international investment position, or total outstanding external debt in US million dollars for a specific quarter, year, or date.
        Sample queries:
        a. What was India's foreign exchange reserves at the end of Q3 2023?
        b. Show the quarterly trends in India's net capital account balance between 2018 and 2022.


        9. top_fifty_macro_economic_indicators_monthly_data: Resolve to this table when the query involves monthly data on India’s macroeconomic indicators such as commercial paper outstanding, net foreign direct investment (FDI), FDI inflows to India, FDI outflows by India, net portfolio investment, total investment inflows, or foreign currency non-resident (FCNR) bank flows. Also resolve here when the user asks about external commercial borrowings (ECB) registrations, exports, imports, or trade balance in US million dollars. Include queries regarding retail payments, digital payments, and market borrowing by state governments in rupee crore, or monthly exchange rate of the Indian rupee against the US dollar. This table should be used for queries analyzing monthly trends in foreign investment, international trade, capital market activity, currency movement, and payment systems performance.
        Sample queries:
        a. What was India’s foreign exchange reserves in July 2024?
        b. Show the trend in the trade balance for India from 2018 to 2025.


        10. top_fifty_macro_economic_indicators_fortnightly_data: Resolve to this table when the query involves fortnightly data related to India's monetary aggregates, banking sector trends, or investment metrics. Trigger this table for keywords like investment in India, aggregate deposits, cash-deposit ratio, credit-deposit ratio, M3 or broad money, and certificates of deposit outstanding.
        Sample queries:
        a. What was the repo rate and currency in circulation in India on January 26, 2024?
        b. Show the trend of broad money (M3) in India for the years 2020 to 2023.


        11. other_macro_economic_indicators_weekly_data: Resolve to this table when the query includes keywords related to call money rate, call money borrowing rate, high/low borrowing rates, short-term borrowing rate, or weekly liquidity conditions in the money market. Also trigger this table for queries involving foreign exchange reserves, forex reserves, FX reserves, or foreign currency assets, particularly when values are expressed in crores. Use this table for weekly trends, date-specific values, or comparative analysis of these metrics over short durations. This table is relevant when the query focuses on monetary policy signals, market liquidity, or external sector strength via reserve and rate indicators captured on a weekly basis.
        Sample queries:
        a. What was the interest rate and monetary aggregate on 8 March 2024 at the all-India level?
        b. How have the reported interest rates and monetary aggregates changed from 2015 to 2025?


        12. other_macro_economic_indicators_quaterly_data: Resolve to this table when the query involves quarterly macroeconomic indicators such as Balance of Payments (BoP) components, current account balance, merchandise trade (exports/imports), services trade, net income, transfers, monetary movements, and errors and omissions, in either INR crore or USD million. Trigger this table when keywords like foreign direct investment (FDI), portfolio investment, capital account balance, BoP credit/debit, external loans, or official/private transfers are mentioned. This table supports analysis of India's external sector health, investment flows, real estate trends, and overall economic output on a quarterly frequency. Use this for any in-depth question involving India's external accounts, capital movements.
        Sample queries:
        a. Show the growth in total bank deposits in India from 2010 to 2024.
        b. What was the broad money aggregate value for July 1, 2023, according to the RBI Database?


        13. other_macro_economic_indicators_monthly_data: Resolve to this table for monthly macroeconomic data involving money supply aggregates (M1, M2, M3), domestic credit, bank credit to government or commercial sector, and detailed RBI balance sheet metrics (like assets, liabilities, loans, and currency circulation). Route queries mentioning fiscal deficit, gross primary deficit, government revenue/expenditure, or interest payments here. Use this table for topics related to foreign trade (e.g., exports/imports in ₹ crore), inflation metrics like Wholesale Price Index (WPI) for all commodities or manufactured products, and BSE market capitalization. Additionally, use it when the question concerns monetary aggregates, credit growth, central bank operations, or public currency holdings. This table enables insights into monetary policy effects, fiscal discipline, trade performance, and financial market trends on a monthly frequency.
        Sample queries:
        a. How did total deposits and advances in Indian banks evolve from 1990 to 2020 on a monthly basis?
        b. What was the total amount of monetary aggregates reported in India in May 2023 and how did it compare to May 2018?


        14. other_macro_economic_indicators_daily_data: Resolve to this table when the query involves daily macroeconomic indicators related to India’s financial markets, especially those requiring high-frequency data. Use this table for queries mentioning NSE Nifty, BSE Bankex, repo rate, reverse repo rate, call money rate (high/low), or the RBI’s USD/INR reference rate. It is appropriate when users ask about market reactions, monetary policy effects, or exchange rate changes on specific dates.
        Sample queries:
        a. What was the USD to INR exchange rate and repo rate on 24 September 2023?
        b. Show Nifty and Sensex values along with bank rates for all available data from 2021.


        15. statewise_nsdp: This table contains statewise Net State Domestic Product (NSDP) data in both constant and current prices with growth rates by state and year. Use for queries about state-level economic performance, NSDP values, and state economic growth.
        Query: "What was the NSDP of Kerala in 2022-23?"
        Query: "Compare Net State Domestic Product growth rates of Tamil Nadu and Karnataka."
        Query: "Show the NSDP trend for Uttar Pradesh over the last 5 years."
        Query: "Which state has the highest NSDP growth rate in 2023-24?"
        Query: "NSDP values at constant prices for Maharashtra from 2018 to 2023."


        16. statewise_nsva: This table contains statewise Net State Value Added (NSVA) data in constant and current prices by state, industry, sub-industry, and economic sectors (primary, secondary, tertiary). Use for queries about sectoral contribution to state economies and industry-wise state performance.
        Query: "Net State Value Added by manufacturing sector in Gujarat."
        Query: "Compare sectoral NSVA between primary, secondary and tertiary sectors in Rajasthan."
        Query: "Agriculture sector contribution to NSVA in Punjab over the last decade."
        Query: "Industry-wise NSVA breakdown for West Bengal in 2022-23."
        Query: "Tertiary sector NSVA growth in Karnataka vs Telangana."


        17. statewise_pcnsdp: This table contains statewise per capita Net State Domestic Product (PCNSDP) data in both constant and current prices with growth rates. Use for queries about per capita income, living standards, and per capita economic performance by state.
        Query: "Per capita NSDP of Goa in 2023-24."
        Query: "Which states have the highest per capita income in India?"
        Query: "Compare per capita NSDP growth rates of Delhi and Mumbai."
        Query: "Show per capita income trends for northeastern states."
        Query: "States with lowest per capita NSDP in the most recent year."


        18. niryat_ite_commodity : This table provides annual commodity-wise export data for India in million USD, with columns for fiscal year, commodity group, total exports, monthly values (Feb/Mar), month-on-month growth %, share %, and update date. This table provides annual export data for India, disaggregated by product category such as 'Electronic Goods', 'Marine Products', 'Ready-made garments', and 'Drugs And Pharmaceuticals'. Data is available for fiscal years like 2023-24 and 2024-25, showing export value, monthly figures, growth rates, and percentage contribution at the national level. The table includes a 'Total' entry for aggregate exports.",
        “What were India top 5 export commodities in FY 2023-24?”
        “Compare petroleum product and electronic goods exports in the last 3 years.”
        “Show the month-on-month growth in gems and jewellery exports for FY 2024-25.”
        “Which commodity group had the highest export share in FY 2022-23?”


        19. niryat_ite_state : This table provides annual state/UT-wise export data for India in million USD, with columns for fiscal year, state/UT, total exports, monthly values (Feb/Mar), month-on-month growth %, share %, and update date. This table provides annual, state-wise financial or production data for Indian states and union territories (e.g., Maharashtra, Kerala, Nagaland, Daman & Diu And Dadra & Nagar Haveli) for financial years such as 2024-25. Each row includes state, year, three quantitative measures, a computed value (possibly percentage change), a ratio or percentage, and a reference date (e.g., 3-Sep-25).
        “Which state exported the most in FY 2022-23?”
        “Rank the top 3 states by export share in FY 2024-25.”
        “Compare Gujarat and Maharashtra exports from FY 2019-20 to FY 2023-24.”
        “What share of India’s exports came from Tamil Nadu in FY 2020-21?”


        20. imf_dm_export : This table contains annual data on a quantitative indicator (e.g., emissions, GDP, or population) for various geographical regions, including individual countries (e.g., Estonia, Brazil, Iran), world regions (South Asia, Europe), and economic groups (Emerging and Developing Asia). The data spans multiple years (e.g., 1982, 2024, 2029). This table provides annual country-wise GDP share (PPP, % of world) from 1980–2029, with columns for country, year, gdp_share_ppp (share of world GDP in PPP terms).
        “What was India’s GDP share in PPP terms in 2022?”
        “List the top 5 countries by share of world GDP in 1980.”
        “How has China’s share of world GDP (PPP) changed from 2000 to 2020?”
        “Which country is projected to have the largest increase in GDP share between 2010 and 2029?”
        “Show the trend of United States vs European Union GDP share from 1980 to 2025.”


        21. gdp_india_fy_primsector_estimates_view : This view provides annual GDP estimates and growth rates for India's PRIMARY Sector, including both constant and current price values.
        Instructions: Use this view to analyze year-wise GDP values and growth rates for the Primary Sector in India, either at constant or current prices.
        Query: Show the GDP values at constant prices for the Primary Sector for all available years.
        Query: List the growth rates at current prices for the Primary Sector from 2011-12 to 2013-14.
        Query: Get the most recent data update date for the Primary Sector GDP estimates.


        22. gdp_india_fy_secsector_estimates_view : This view provides annual GDP estimates for India's SECONDARY sector, including values at constant and current prices, along with their respective growth rates.
        Instructions: Use this view to analyze year-wise GDP figures and growth rates for the secondary sector in India, either at constant or current prices.
        Query: Show the secondary sector GDP values at constant prices for each year.
        Query: What was the growth rate at current prices for the secondary sector in 2013-14?
        Query: List all years with their corresponding GDP values and growth rates for the secondary sector.


        23. gdp_india_fy_tersector_estimates_view : This view provides annual GDP estimates for India's tertiary sector, including values at constant and current prices, and their respective growth rates.
        Instructions: Use this view to analyze year-wise GDP figures and growth rates for the tertiary sector in India, either at constant or current prices.
        Examples:
        Query: Show the GDP value and growth rate for the tertiary sector in 2013-14.
        Query: List all years with their corresponding constant price GDP values for the tertiary sector.
        Query: What was the growth rate at current prices for the tertiary sector in 2012-13?


        24.gdp_india_fy_primsector_estimates_dtls_view : This table/view provides annual GDP estimates and growth rates for India's primary sector sub-components (like agriculture, livestock, forestry, and fishing), with values at both constant and current prices.
        Instructions: Use this table/view to retrieve year-wise GDP values and growth rates for specific named primary sector items, at constant or current prices.
        Examples:
        Query: Show the GDP at constant prices for agriculture, livestock, forestry & fishing for each year.
        Query: List the growth rates at current prices for all primary sector items in 2013-14.
        Query: Get the most recent data update date for this table.


        25. gdp_india_fy_secsector_estimates_dtls_view : This table provides annual GDP estimates for India's secondary sector sub-components, including values and growth rates at both constant and current prices.
        Instructions: Use this table to analyze year-wise GDP values and growth rates for specific secondary sector items (manufacturing, construction, and utilities) in India, at both constant and current prices. Filter by 'item' for sub-sector details and by 'year' for time-based analysis.
        Query: Show the manufacturing GDP and its growth rate for each year.
        Query: List all secondary sector items with their GDP values for 2013-14.
        Query: What was the growth rate at constant prices for each secondary sector item in 2012-13?


        26. gdp_india_fy_tersector_estimates_dtls_view : This table/view provides annual GDP estimates for various tertiary sector items in India, including values at constant and current prices, along with their growth rates.
        Instructions: Use this table/view to analyze GDP contributions and growth rates for specific tertiary sector items (financial services, public administration, trade) by year, at both constant and current prices.
        Query: Show the GDP value at constant prices for 'TRADE, HOTELS, TRANSPORT, COMMUNICATION & SERVICES RELATED TO BROADCASTING' in 2012-13.
        Query: List the growth rates at current prices for all tertiary sector items in 2013-14.
        Query: Get the GDP values at current and constant prices for each year for 'TRADE, HOTELS, TRANSPORT, COMMUNICATION & SERVICES RELATED TO BROADCASTING'.


        27. gdp_india_fy_expenditure_estimates_dtls_view : This view provides annual GDP expenditure estimates for India, including values and growth rates (both constant and current prices) for different expenditure items such as GFCE, with data updated as of the latest available date.
        Instructions: Use this view to retrieve GDP expenditure data for India by year and expenditure item, including values and growth rates at both constant and current prices.
        Query: Show the constant price GDP expenditure values for all items in 2012-13.
        Query: List the growth rates at current prices for GFCE from 2011-12 to 2013-14.
        Query: Get all available data for the year 2013-14.


        28. gdp_india_qtr_estimates_view : This view provides quarterly GDP estimates for India, including values at constant and current prices, along with their respective growth rates, for each year and item where the GDP flag is true.
        Instructions: Use this view to retrieve quarterly (Q1, Q2, Q3, Q4) GDP data for India, including values and growth rates at both constant and current prices, filtered for records marked as GDP.
        Query: Show the quarterly GDP values at constant prices for the year 2011-12.
        Query: Get the quarterly growth rates at current prices for all years.
        Query: Show the GDP growth for the last 8 quarters.


        29. gdp_india_qtr_primsector_estimates_view : This view provides quarterly GDP estimates for India's primary sector, including values at constant and current prices, along with their respective growth rates.
        Instructions: Use this view to analyze GDP trends, values, and growth rates for the primary sector (like agriculture, livestock, forestry, and fishing) in India by year and quarter. Filter by 'year' or use the 'data_updated_date' to get the latest data.
        Query: Show the constant price GDP values for the primary sector in 2015-16.
        Query: Get the growth rates at current prices for the primary sector for all available years.


        30. gdp_india_qtr_secsector_estimates_view : This view provides quarterly GDP estimates for India's secondary sector, including values at constant and current prices, along with their respective growth rates.
        Instructions: Use this view to analyze quarterly GDP data specifically for the secondary sector (manufacturing, construction, and utilities) in India, focusing on values and growth rates at both constant and current prices. Filter by year or other columns as needed.
        Query: Show the constant price GDP values for the secondary sector for the year 2015-16.
        Query: List the growth rates at current prices for each quarter in 2018-19.
        Query: Get all available data for the secondary sector for the year 2020-21.


        31.gdp_india_qtr_tersector_estimates_view : This view provides quarterly GDP estimates for India's tertiary sector, including values at constant and current prices, growth rates, and data update dates.
        Instructions: Use this view to analyze GDP figures and growth rates for the tertiary sector (financial services, public administration, trade) in India by year and quarter, with both constant and current price values.
        Query: Show the constant price GDP values for the tertiary sector in 2011-12.
        Query: List the growth rates at current prices for each year in the tertiary sector.


        32. gdp_india_qtr_primsector_estimates_dtls_view : This view provides quarterly GDP estimates for India's primary sector, detailing values and growth rates for specific items like agriculture and mining, at both constant and current prices.
        Instructions: Use this view to analyze or compare GDP figures and growth rates for primary sector components (excluding the overall 'PRIMARY SECTOR') by year and item, at constant or current prices.
        Query: Show the constant price GDP values for agriculture for each year.
        Query: Get the latest GDP value for mining sector.


        33. gdp_india_qtr_secsector_estimates_dtls_view : This view provides quarterly GDP estimates for India's secondary sector, detailing values and growth rates (both constant and current prices) for sub-sectors like manufacturing, construction, and utilities.
        Instructions: Use this view to analyze or retrieve GDP data for secondary sector sub-industries in India by year, including their values and growth rates at both constant and current prices. Filter by 'year' or 'item' as needed.
        Query: Show the constant price GDP values for manufacturing from 2015-16 onwards.
        Query: List the growth rates at current prices for all secondary sector items in 2018-19.
        Query: Get the latest updated GDP values (current and constant) for construction.


        34.gdp_india_qtr_tersector_estimates_dtls_view : This view provides quarterly GDP estimates for India's tertiary sector, detailing values and growth rates (both constant and current prices) for specific service-related items.
        Instructions: Use this view to analyze or retrieve GDP data for individual tertiary sector items (such as financial services, public administration, trade, etc.) by year, including their values and growth rates at both constant and current prices.
        Query: Show the constant price GDP values for all tertiary sector items in 2015-16.
        Query: List the growth rates at current prices for 'FINANCIAL, REAL ESTATE & PROFESSIONAL SERVICES' across all years.
        Query: Get all available data for 'TRADE, HOTELS, TRANSPORT, COMMUNICATION & SERVICES RELATED TO BROADCASTING' for the year 2018-19.


        35. gdp_india_qtr_expenditure_estimates_dtls_view : This view provides quarterly GDP expenditure estimates for India, including values and growth rates (both constant and current prices) for various expenditure items by year.
        Instructions: Use this view to analyze GDP expenditure components, their values, and growth rates for different years. Filter by 'year' or 'item' to focus on specific periods or expenditure categories.
        Query: Show the constant price value and growth rate for 'EXPORTS OF GOODS AND SERVICES' in 2011-12.
        Query: List all items and their current price values for the year 2011-12.
        Query: Get the growth rates at current prices for all items updated on 16-07-2025.


        36. gdp_state_fy_subindustry_actuals_view : This view provides annual GDP figures at constant and current prices for each sub-industry within states, including classification flags and product tax/subsidy indicators.
        Columns: id, base_year, state, industry, sub_industry, primary_flag, secondary_flag, tertiary_flag, taxes_on_products, subsidies_on_products, year, constant_value_in_lakh, current_value_in_lakh, released_on, data_source, updated_on
        Instructions: Use this view to retrieve state-wise, industry-wise, and sub-industry-wise GDP data for specific years, including constant and current values, and to filter by economic sector or product tax/subsidy status.
        Query: Show the constant and current GDP values for 'Crops' in Andaman Nicobar for all available years.
        Query: List all sub-industries under 'Agriculture, forestry and fishing' for the year 2013-14 in Andaman Nicobar.
        Query: Get the GDP values for all primary sector sub-industries in Andaman Nicobar for 2011-12.

        37. none_of_these: for any queries which are unrelated to GDP. Queries regarding the general state of the economy, government policies, and upcoming challenges also fall under the none_of_these category.

        # Consider the list above, and respond ONLY with one of the file names from the following list:
        [india_fy_gdp_components_view, india_fy_national_income_view, gdp_state_fy_actuals, gdp_state_fy_industry_actuals_view, per_capita_income_product_final_consumption,
        gdp_india_fy_estimates_view, gdp_india_fy_primsector_estimates_view, gdp_india_fy_secsector_estimates_view, gdp_india_fy_tersector_estimates_view,
        other_macro_economic_indicators_daily_data, other_macro_economic_indicators_monthly_data, other_macro_economic_indicators_quaterly_data,other_macro_economic_indicators_weekly_data,
        top_fifty_macro_economic_indicators_monthly_data,top_fifty_macro_economic_indicators_quaterly_data,top_fifty_macro_economic_indicators_weekly_data,
        statewise_nsdp, statewise_nsva, statewise_pcnsdp, top_fifty_macro_economic_indicators_fortnightly_data,
        niryat_ite_state, niryat_ite_commodity, imf_dm_export, gdp_india_fy_primsector_estimates_dtls_view,
        gdp_india_fy_secsector_estimates_dtls_view, gdp_india_fy_tersector_estimates_dtls_view, gdp_india_fy_expenditure_estimates_dtls_view, gdp_india_qtr_estimates_view,
        gdp_india_qtr_primsector_estimates_view, gdp_india_qtr_secsector_estimates_view, gdp_india_qtr_tersector_estimates_view, gdp_india_qtr_primsector_estimates_dtls_view,
        gdp_india_qtr_secsector_estimates_dtls_view, gdp_india_qtr_tersector_estimates_dtls_view, gdp_india_qtr_expenditure_estimates_dtls_view,gdp_state_fy_subindustry_actuals_view, none_of_these]
        DO NOT include any reasoning traces or other text apart from the file name selected from the above list.
""")
    selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_IIP(query):
    system_instruction=dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        # IMPORTANT RULES:
        - For any queries LONGER THAN 2 years duration, pick ANNUAL or YEARLY tables where available.
        - If the query is about cement or construction-related indicators or pm schemes like pmay, pmgsy, use the construct_state_cement_indicators_view.
        - If the query is BROAD or GENERAL (e.g. just "IIP growth", "IIP trends", "overall industrial performance") — choose the corresponding *category_view* file (monthly or yearly), not the subcategory-specific files.
        - If the query is STATE-SPECIFIC and mentions "Assam" or industries in Assam, then use iip_in_assam.


        # Table information

        1. construct_state_cement_indicators_view : This view provides state-wise indicators related to cement and construction, including housing scheme progress (PMAY-G, PMAY-U), road construction (PMGSY, Bharatmala), limestone resources and production, cement and power capacities, economic value additions, real estate, and population for each state as of a specific date.
        Columns: state, pmay_g_target_households, pmay_g_target_households_num, pmay_g_completed_households, pmay_g_completed_households_num, pmay_u_target_households, pmay_u_target_households_num, pmay_u_completed_households, pmay_u_completed_households_num, pmgsy_road_length_sanctioned_km, pmgsy_road_length_completed_km, bharatmala_road_length_targeted_km, bharatmala_road_length_completed_km, limestone_total_resources_kt, limestone_proved_reserve_kt, limestone_production_kt, installed_cement_capacity_mtpa, captive_power_capacity_mw, whrs_capacity_mw, gsdp_inr_crore, mining_value_addition_inr_crore, manufacturing_value_addition_inr_crore, construction_value_addition_inr_crore, real_estate_inr_crore, total_population_2024, date_stamp
        Instructions: Use this view to analyze or compare states on construction, cement industry, infrastructure progress, limestone resources, and related economic indicators. Filter by state or date_stamp for the latest or historical data.
        Query: Show the total installed cement capacity and limestone production for each state.
        Query: Which states have completed more than 1,000,000 PMAY-U households?
        Query: List construction value addition for Andhra Pradesh.
        Query: Find states with limestone total resources above 10 million tonnes.
        Query: Get the latest PMGSY road length completed for all states.


        2. iip_india_yr_catg_view : This table/view provides annual Index of Industrial Production (IIP) data for India, categorized by sector and category, including index values and growth rates for each year.
        Instructions: Use this table/view to retrieve yearly IIP index and growth rate data for India, filtered by base year, sector type, category, or year as needed.
        Query: Show the IIP index and growth rate for all categories in 2024-25.
        Query: Get the manufacturing sector's IIP index for the base year 2011-12.
        Query: List all available years and their general IIP index values.


        3.iip_india_mth_catg_view : This table/view provides monthly Index of Industrial Production (IIP) data for India, categorized by sector type and category, including index values and growth rates, along with metadata such as data source, release dates, and fiscal year.
        Instructions: Use this table/view to retrieve IIP index and growth rate data for specific months, years, sector types, or categories. Filter by 'year', 'month', 'sector_type', 'category', or 'fiscal_year' as needed to analyze industrial production trends.
        Query: Show the IIP index and growth rate for all categories in June 2023.
        Query: Get the IIP growth rate for 'Consumer Durables' in April 2023.
        Query: Monthly trends of IIP in 2019-20.
        Query: Find the IIP index for the 'Mining' category in July 2019.


        4. iip_india_yr_subcatg_view : This view provides annual Index of Industrial Production (IIP) data for India, broken down by sector, category, and sub-category, including index values and growth rates.
        Instructions: Use this view to retrieve yearly IIP indices and growth rates for specific sectors, categories, or sub-categories, filtered by year, base year, or other relevant attributes.
        Query: Show the IIP index and growth rate for all manufacturing sub-categories in 2024-25.
        Query: List all available years for which IIP data is present for the 'Manufacture of Textiles' sub-category.
        Query: Get the latest updated IIP index for each sub-category under the Manufacturing category.


        5. iip_india_mth_subcatg_view : Monthly Index of Industrial Production (IIP) data for India, broken down by sector, category, and sub-category, including index values and growth rates.
        Instructions: Use this table/view to retrieve monthly IIP index and growth rate data for specific sectors, categories, or sub-categories, filtered by year, month, or fiscal year as needed.
        Query: Show the IIP index and growth rate for 'Manufacture of Rubber and Plastics Products' in July 2023.
        Query: Month-on-month trends for every industry sector, category, sub-sector since 2023.


        6. iip_in_assam: This file contains Index of Industrial Production (IIP) data specifically for Assam on an annual basis, broken down by industry NIC codes, descriptions, and weights. Use this table if the query is about Assam or state-specific industrial production.
        Sample queries:
        a. What was the Index of Industrial Production for Assam in 2018-19?
        b. Show the IIP for Manufacture of Food Products in Assam from 2012-13 to 2017-18.
        c. Give the industry-wise IIP for Assam in 2016-17.
        d. Trend of IIP in Assam across the last 5 years.


        7. iip_in_andra_pradesh_sector_wise : This table provides the Index of Industrial Production (IIP) for Andhra Pradesh at the sectoral level (e.g., Mining & Quarrying, Manufacturing, Electricity). It is reported monthly and grouped by fiscal year. Use this for broader sector level queries in AP.
        “Show the IIP trend for Andhra Pradesh Mining & Quarrying in 2022–23.”
        “Which sector had the highest index value Andhra Pradesh in July 2023?”
        “Compare the average IIP for Manufacturing Andhra Pradesh between 2021–22 and 2022–23.”
        “What is the IIP of Electricity Andhra Pradesh sector in April 2024?”


        8. iip_in_andra_pradesh_sector_industry_wise : This table provides industry-wise IIP values in Andhra Pradesh, classified by NIC code, industry description, weight, year, and month. It allows analysis at a more granular level within sectors. Use this for specific questions on industries in AP.
        “What was the index value for Manufacture of Textiles in Andhra Pradesh June 2024?”
        “List the top 5 industries by index value in Andhra Pradesh June 2024.”
        “Compare the performance of Manufacture of Food Products vs Manufacture of Beverages in Andhra Pradesh 2024–25.”
        “Which industry showed the largest growth in index value Andhra Pradesh from May to June 2024?”


        9. iip_in_andra_pradesh_use_wise : This table provides the IIP for Andhra Pradesh by use-based classification (e.g., Primary Goods, Capital Goods, Intermediate Goods, Consumer Goods). It is reported monthly across fiscal years. Use this for queries on the consumption side of industries in AP.
        “Show the IIP trend for Primary Goods in Andhra Pradesh 2023–24 vs 2024–25.”
        “What was the Capital Goods index value in Andhra Pradesh April 2024?”
        “Compare the average IIP of Consumer Durables vs Consumer Non-Durables in Andhra Pradesh 2022–23.”
        “Which use-based category Andhra Pradesh had the sharpest decline between April and June 2024?”


        10. iip_in_rajasthan_monthly : This table provides the monthly Index of Industrial Production (IIP) for Rajasthan, categorized into General, Manufacturing, Electricity, and Mining, by fiscal year and month.
        “Show the monthly IIP trend for Manufacturing in Rajasthan 2025–26.”
        “What was the Electricity index in Rajasthan May 2025?”
        “Compare General index values Rajasthan for April, May, and June of 2025–26.”
        “Which category had the lowest index value in Rajasthan June 2025?”


        11. iip_in_rajasthan_fy_index : This table provides the annual average IIP for Rajasthan across categories such as Electricity, Manufacturing, Mining, and the Total General Index, by fiscal year.
        “What was the Total General Index in Rajasthan 2021–22 vs 2022–23?”
        “Which sector showed the highest growth Rajasthan between 2021–22 and 2022–23?”
        “List the average IIP values for Rajasthan Manufacturing across all fiscal years.”
        “Compare the Electricity index Rajasthan between 2021–22 and 2022–23.”


        12. iip_in_rajasthan_two_digit_index : This table provides the IIP for Rajasthan by two-digit NIC industry codes, with industry descriptions and fiscal year index values.
        “What was the IIP value for Manufacture of Textiles in Rajasthan 2025–26?”
        “List the top 5 industries by index value in Rajasthan 2025–26.”
        “Compare Manufacture of Food Products and Manufacture of Beverages in Rajasthan 2025–26.”
        “Which industry had the lowest index value in Rajasthan 2025–26?”


        13. iip_in_kerala_fy_index : This table provides the annual Index of Industrial Production (IIP) for Kerala, reported by fiscal year and category (e.g., Manufacturing), along with corresponding index values.
        “What was Kerala’s Manufacturing index in Kerala 2019–20?”
        “Compare the Manufacturing IIP Kerala between 2018–19 and 2020–21.”
        “Show the year-wise trend of Kerala’s Manufacturing IIP from 2015–16 to 2023–24.”
        “In which year did the Manufacturing index reach its lowest value in Kerala?”


        14. iip_in_kerala_monthly : This table provides monthly IIP data for Kerala by fiscal year, category (e.g., General, Manufacturing, Electricity), and month, with corresponding index values.
        “What was the General IIP in Kerala April 2020–21?”
        “Compare April values of General Kerala IIP across fiscal years 2018–19 to 2023–24.”
        “Show the month-on-month trend of General index in Kerala 2019–20.”
        “Which fiscal year had the highest April General index in Kerala?”


        15. iip_in_kerala_quarterly : This table provides quarterly IIP data for Kerala by fiscal year, quarter (Q1, Q2, Q3, Q4), and category (e.g., Manufacturing), with corresponding index values.
        “What was Kerala’s Manufacturing index in Kerala Q2 of 2016–17?”
        “Compare Q1 Manufacturing Kerala values across fiscal years 2015–16 to 2017–18.”
        “Show the quarter-wise Manufacturing Kerala IIP trend for 2016–17.”
        “Which quarter had the highest Manufacturing Kerala index in 2015–16?”

        16. annual_chemical_production_data: This table contains annual production data for various chemical products, including group name, product name, year, and production value.
        Instructions: Use this table to retrieve or analyze annual production figures for specific chemical products or groups, filter by year, or aggregate production values.
        Example queries:
        Query: Show the total production value of all Alkali Chemicals in 2014-2015.
        Query: List the production values for each product in 2014-2015.
        Query: Get the annual production of Caustic Soda for all available years.

        17. vehicle_registrations_state: This table contains the number of vehicle registrations in each Indian state and union territory for the years 2021 to 2025, along with the total registrations over these years.
        Instructions: Use this table to analyze or compare vehicle registration counts by state and year, or to find total registrations for specific states or time periods.
        Example queries:
        Query: Show the total vehicle registrations for each state.
        Query: Which state had the highest number of vehicle registrations in 2024?
        Query: List the vehicle registrations in Andhra Pradesh for each year from 2021 to 2025.
        Query: Show the total vehicle registrations in 2023 across all states.

        18. none_of_these: for any queries which are unrelated to IIP. Queries regarding the general state of the economy, government policies, and upcoming challenges also fall under the none_of_these category.

        Consider the list above, and respond ONLY with one of the file names from the following list:
        [construct_state_cement_indicators_view, iip_india_yr_catg_view, iip_india_mth_catg_view,iip_india_yr_subcatg_view,iip_india_mth_subcatg_view,iip_in_assam,iip_in_andra_pradesh_sector_wise,iip_in_andra_pradesh_sector_industry_wise,iip_in_andra_pradesh_use_wise,
        iip_in_rajasthan_monthly,iip_in_rajasthan_fy_index,iip_in_rajasthan_two_digit_index,iip_in_kerala_fy_index,iip_in_kerala_monthly,iip_in_kerala_quarterly, annual_chemical_production_data, vehicle_registrations_state, none_of_these]
        DO NOT include any reasoning traces or other text apart from the file name selected from the above list.
    """)
    selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_MSME(query):
    system_instruction=dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        1. msme_gbc_food_non_food_view : This table contains Monthly Gross Bank Credit (GBC) outstanding, grouped by category (Food credit / Non-Food credit),the column 'gbc_in_cr' holds the amount of Gross Bank Credit (in crores).
        Query1: "What was the Food Gross Bank Credit on March 2020?"
        Query2: "Compare Non-Food and Food GBC for March 2021."
        Query3: "Show the monthly trend of Non-Food GBC for the year 2021."
        Query4: "How did Food Credit GBC change from Jan 2020 to Dec 2021?"
        Query5: "What is the overall Gross Bank Credit trend over last 5 years?"


        2. msme_definitions_by_sector : This table defines the criteria used to classify MSMEs (Micro, Small, and Medium Enterprises) in different Asian countries based on sector (Manufacturing, Services) and criteria like employees, annual income, or annual turnover.  It specifies the thresholds for each category (Micro, Small, Medium), for international MSME classification comparisons.
        Sample queries:
        a. What are the thresholds for number of employees to qualify as an SME in manufacturing in Vietnam?
        b. How does the annual turnover criteria for services and other sectors differ between Tajikistan and Malaysia?


        3. msme_state_ureg_recent : This table contains data on MSME registrations in India, broken down by state. It includes the number of micro, small, and medium enterprises registered, as well as the total number of Udyam registrations and total MSMEs.
        Sample queries:
        a. Which state had the highest number of micro, small, and medium enterprises registered as of July 21, 2025?
        b. What is the total number of MSME enterprises and employment recorded in Odisha according to this dataset?


        4.msme_gbc_non_food_dtl_view : This table contains Monthly Gross Bank Credit (GBC) outstanding by economic sector for non food credit (e.g., Agriculture, Services,Personal Loans,Industries etc.), the column 'gbc_in_cr' holds the amount of Gross Bank Credit (in crores).
        Query1: "What was the Gross Bank Credit to the Agriculture sector in March 2020?"
        Query2: "Show the Personal Loans GBC values across all months of 2021."
        Query3: "How much GBC was given to the Services sector in April 2021?"
        Query4: "List sector-wise GBC on 27 March 2020."
        Query5: "Which sector had the highest GBC in 2021? Provide effective and release dates."


        5. nifty_sme_index_daily_values : Use this table when the question is for NIFTY SME INDEX for Nifty related data .
        Sample queries:
        a. Show the yearly average value of NIFTY SME EMERGE index from 2017 to 2022.
        b. On which date did the NIFTY SME EMERGE index surpass 10,000 points for the first time?


        6.msme_share_by_region_view : Use this table when the question asks for MSME share in the economy grouped by region, subregion, or country.It contains MSME share percentage in the economy, by region/subregion/country
        Query1: "What is the MSME share in Pacific Islands in 2021?"
        Query2: "MSME share trend in South Asia."


        7.msme_share_by_sector_view : Use this table when the question asks for MSME sharebroken down by sector in any region or country.It contains MSME share percentage by sector (e.g., services, manufacturing) across Asia
        Query1: "MSME share in manufacturing sector in Sri Lanka?"
        Query2: "Which sector had the highest MSME share in Southeast Asia in 2020?"


        8.msme_priority_sector_view : This table contains the gross bank credit, GBC outstanding in crores for the top ten most important and in priority sectors among all non food sectors in the column outstanding_as_on
        a. "Give the priority sectors outstanding value for the years of 2020 and 2021."
        b. What was the total value of priority sector lending for Micro and Small Enterprises in March 2023?
        c. Show the trend of Export Credit category at the all-India level from 2021 to 2024.


        9.msme_industry_view : This table contains the gross bank credit GBC outstanding in crores for all subgroups of the industry sector such as construction, food processing etc., in the column outstanding_as_on
        Query1: "Give the textile industry's outstanding value for the years of 2020 and 2021."


        10. msme_global_view : This table contains country-level, annual data for various Asian and Pacific nations (e.g., India, Indonesia, Malaysia, Georgia, Fiji), covering economic, population, and education indicators such as employment, labor force, GDP, school enrollments, and literacy rates. Data spans multiple years, features diverse metrics (e.g., school-age population, GDP, enrollment rates), and includes subregional groupings (Central and West Asia, Southeast Asia, South Asia, Pacific Islands). Data sources include the Asian Development Bank/ERDI. Contains aggregated global MSME (Micro, Small & Medium Enterprises) statistics per region, country, and year. Data includes MSME counts, employee counts, GDP contributions, exports, ratios, exchange rate etc.
        a. "What was the primary school enrollment in India and Indonesia in 2017?",
        b. "Provide the school-age population and literacy rate for Malaysia and Vietnam for the year 2019."
        c. "What is the total number of MSMEs in India in 2020?"
        d. "Compare GDP contribution of MSMEs between USA and China in 2019."
        e. "Show me the trend of employee count in the ASEAN region over the last 5 years."
        f. "Which country had the highest export ratio among MSMEs in 2021?"


        11. msme_india_mth_sector_bankcredit_view : Monthly outstanding bank credit data in India by sector, including MSME, personal loans, and agriculture, sourced from the Reserve Bank of India. SECTOR LEVEL
        Instructions: Use this view to analyze monthly outstanding bank credit amounts by sector, year, and month. Filter by 'sector', 'year', or 'month' as needed to get specific credit data. The 'outstanding_as_on' column provides the credit amount (in crore) as of the effective date.
        Query: Show the outstanding bank credit for MSME sectors in July 2019.
        Query: List the outstanding credit for all sectors for the fiscal year 2019-20.
        Query: Get the total outstanding bank credit for 'Personal Loans' in 2019.


        12. msme_india_mth_grp_bankcredit_view : This view provides monthly outstanding bank credit data for different MSME-related sectors and groups in India, sourced from the Reserve Bank of India. It includes details such as year, month, sector, group name, outstanding amount, and relevant dates.
        Instructions: Use this view to analyze trends, compare outstanding credit amounts, or extract time-series data for specific MSME sectors or groups by month and year.
        GROUP LEVEL
        Query: Show the total outstanding bank credit for the 'Services' sector in fiscal year 2019-20.
        Query: List the monthly outstanding amounts for 'Professional Services' group in 2019.
        Query: Get the latest outstanding amount for each sector as of March 2020.


        13. msme_india_mth_subgrp_bankcredit_view : This view provides monthly outstanding bank credit data for various MSME sectors and subgroups in India, including details like sector, group, subgroup, outstanding amount, and reporting dates.
        SUBGROUP LEVEL
        Instructions: Use this view to analyze trends in bank credit outstanding amounts for different MSME sectors, groups, and subgroups across months and years. Filter by year, sector, group_name, or subgroup to get specific insights.
        Query: Show the outstanding bank credit for 'Power' under 'Infrastructure' group for January 2021.
        Query: List all subgroups and their outstanding amounts for the 'Trade' group in April 2019.
        Query: Get the total outstanding bank credit for 'Industry Micro Small Medium Large' sector in fiscal year 2020-21.


        14. upi_dly_stats : This table contains daily statistics of UPI transactions, including transaction volume (in millions) and value (in crores), along with date and metadata. DAILY resolution
        Instructions: Use this table to analyze daily UPI transaction trends, volumes, and values by date, month, or year. Filter by date_stamp, year, or month as needed.
        Query: Show the total UPI transaction volume in May 2021.
        Query: List daily UPI transaction values for the first week of May 2021.
        Query: Get the average daily UPI transaction volume for 2021.


        15.upi_mth_stats : This table contains monthly statistics of UPI transactions in India, including transaction volume (in millions) and value (in crores), along with release and update dates. MONTHLY resolution
        Instructions: Use this table to analyze trends in UPI transaction volumes and values by month and year, or to retrieve data for specific periods. Filter by year, month, or other columns as needed.
        Query: Show the total UPI transaction volume for the year 2025.
        Query: List the UPI transaction value for each month in 2025.
        Query: Find the month with the highest UPI transaction value in 2025.
        Query: Get all months where the transaction volume exceeded 19,000 million.


        16. upi_mth_failures : This table contains monthly UPI transaction failure statistics for various issuer banks, including transaction volumes and percentages of approved, business declined, and technical declined transactions.
        Instructions: Use this table to analyze UPI transaction performance and failure rates by bank, month, or year. Filter by issuer_bank_name, year, or month to get specific data.
        Query: Show the approved percentage for State Bank of India in August 2021.
        Query: List total transaction volumes and failure rates for all banks in August 2021.
        Query: Find the bank with the highest business declined percentage in August 2021.

        17. msme_sambandh_procurement_data: This table contains procurement data by various ministries, including targets and achievements for total procurement, MSEs, SC/ST MSEs, and women MSEs, for each fiscal year.
        Instructions: Use this table to analyze ministry-wise procurement targets and achievements, especially for MSEs, SC/ST MSEs, and women MSEs, across different fiscal years.
        Example queries:
        Query: Show the total procurement achievement for each ministry in 2020-21.
        Query: List ministries where achievement for women MSEs exceeded the target in 2020-21.
        Query: What was the total target and achievement for SC/ST MSEs across all ministries in 2020-21?

        18. rbi_india_mth_payment_system_indicators: Monthly indicators of payment systems in India, including settlement systems and their transaction volumes and values.
        Instructions: Use this table to analyze monthly trends, volumes, and values of different payment and settlement systems in India, filtered by section, item, sub_item, month, or year as needed.
        Example queries:
        Query: Show the total transaction value for CCIL Operated Systems in 2025.
        Query: List the monthly volume of Govt. Securities Clearing for July and August 2025.
        Query: Get all payment system indicators for August 2025.

        19. rbi_india_mth_bank_rtgs: This table contains monthly RTGS (Real Time Gross Settlement) transaction statistics for Indian banks, including inward and outward transaction volumes and values, split by interbank and customer transactions.
        Instructions: Use this table to analyze RTGS transaction data for Indian banks by month and year, including breakdowns by bank, transaction type (inward/outward), and customer/interbank splits.
        Example queries:
        Query: Show the total RTGS inward and outward transaction volumes for all banks in August 2025.
        Query: List the top 5 banks by inward RTGS transaction value in August 2025.
        Query: Get the RTGS outward customer transaction volume for 'AIRTEL PAYMENTS BANK LTD.' in August 2025.
        Query: Find the percentage share of inward RTGS volume for each bank in August 2025.

        20. rbi_india_mth_bank_neft: This table contains monthly NEFT transaction statistics for Indian banks, including counts and amounts for received credits and outward debits.
        Instructions: Use this table to analyze NEFT transaction volumes and values by bank, month, and year. Filter by 'bank_name', 'month', or 'year' to get specific data.
        Example queries:
        Query: Show the total NEFT received amount for all banks in August 2025.
        Query: List the top 5 banks by outward NEFT debits amount in 2025.
        Query: Get NEFT inward credits count and amount for 'ABHYUDAYA CO-OP BANK LTD' for August 2025.

        21. rbi_india_mth_bank_mobile_banking: This table contains monthly mobile banking statistics for Indian banks, including transaction volumes, values, and the number of active mobile banking customers.
        Instructions: Use this table to analyze mobile banking activity by bank, month, or year, such as total transactions, values, or customer counts.
        Example queries:
        Query: Show the total mobile banking transaction value for all banks in August 2025.
        Query: List the top 5 banks by number of active mobile banking customers in 2025.
        Query: Get the monthly mobile banking transaction volume for 'A. P.MAHESH CO-OPERATIVE URBAN BANK LTD.' in 2025.

        22. rbi_india_mth_bank_internet_banking → This table contains monthly data on internet banking transactions for various banks in India, including transaction volumes, values, and the number of active internet banking customers.
        Instructions: Use this table to analyze internet banking trends, compare banks, or aggregate transaction data by month, year, or bank.
        Example queries:
        Query: Show the total internet banking transaction volume for each bank in August 2025.
        Query: List the top 5 banks by value of internet banking transactions in 2025.
        Query: Find the total number of active internet banking customers across all banks for August 2025.

        Note: If a query is about gdp of msme do not select any table return "none_of_these".

        Consider the list above, and respond ONLY with one of the file names from the following list:
        [msme_gbc_food_non_food_view, msme_definitions_by_sector, msme_state_ureg_recent, msme_gbc_non_food_dtl_view, nifty_sme_index_daily_values , msme_share_by_region_view, msme_share_by_sector_view,
        msme_priority_sector_view, msme_industry_view, msme_global_view, msme_india_mth_sector_bankcredit_view, msme_india_mth_grp_bankcredit_view, msme_india_mth_subgrp_bankcredit_view, upi_dly_stats,
        upi_mth_stats, upi_mth_failures, msme_sambandh_procurement_data, rbi_india_mth_payment_system_indicators, rbi_india_mth_bank_rtgs, rbi_india_mth_bank_neft, rbi_india_mth_bank_mobile_banking, rbi_india_mth_bank_internet_banking, none_of_these]
        DO NOT include any reasoning traces or other text apart from the file name selected from the above list.
            """)
    selected_file, i_tokens, o_tokens = llm_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_district_level(query):
    system_instruction = dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        1. youthpower_district_level_metrics : 
           This table stores district-level quantitative indicators for youth empowerment across India, measuring education, employment, entrepreneurship, skills, and socio-economic opportunity. 
           It provides composite scores (Youth Power, Opportunity, Workforce, Education, Readiness & Skills) and a wide range of supporting metrics covering demography, MSME presence, employment structure, CSR spending, education quality, and vocational training intensity.

           Below is a structured summary of its contents and the type of analyses it supports:

           **Category: Geography**
           - Columns: state, district
           - Description: Administrative identifiers for regional data aggregation and comparison.
           - Example Query: "Show all districts in Madhya Pradesh with their youth power scores for 2024."

           **Category: Time**
           - Columns: year, date_stamp, released_on, updated_on
           - Description: Temporal indicators to filter data by year or release cycle.
           - Example Query: "Compare the youth power scores between 2023 and 2024 for all districts in Bihar."

           **Category: Demographics**
           - Columns: total_population_in_lacs, total_youth_population_in_lacs
           - Description: Demographic indicators of total and youth population size.
           - Example Query: "List the top 10 districts by total youth population in 2024."

           **Category: Economic Activity**
           - Columns: number_of_jobs_in_lacs, registered_unorganised_workers_in_lacs, labor_force_participation_percent, unemployment_rate, epfo_coverage_rate
           - Description: Indicators for labor force structure, employment, and worker coverage.
           - Example Query: "Find districts where the unemployment rate exceeds 12% and EPFO coverage is below 50% in 2024."

           **Category: MSME Indicators**
           - Columns: msmes_per_10k_population, msme_micro_percent, msme_small_percent, msme_medium_percent, manufacturing_enterprises_percent, services_enterprises_percent
           - Description: Distribution and composition of MSMEs across districts.
           - Example Query: "Show the top 5 districts with the highest MSMEs per 10k population in Maharashtra."

           **Category: MSME Employment**
           - Columns: msme_micro_employment_percent, msme_small_employment_percent, msme_medium_employment_percent
           - Description: Employment share by MSME size category.
           - Example Query: "List districts where micro enterprises contribute over 60% of MSME employment."

           **Category: Financial Inclusion**
           - Columns: savings_per_working_age_in_lacs, mudra_loan_to_labour_force_ratio_in_thousands, csr_spending_per_capita_in_rupees, csr_share_percent
           - Description: Indicators of access to finance, credit, and CSR activities.
           - Example Query: "Find districts where CSR spending per capita exceeds ₹200 in 2024."

           **Category: Infrastructure**
           - Columns: trains_per_week_per_1000_sqkm
           - Description: Proxy for transport connectivity and accessibility.
           - Example Query: "Show the top 10 districts with the highest train frequency per 1000 sq km."

           **Category: Education & Skills**
           - Columns: number_of_schools_in_thousands, private_schools_percent, vocational_schools_percent, enrollment_ratio, ger_class_6_to_8, ger_class_9_to_12, test_scores_percent, english_score_class_10, maths_score_class_10, number_of_colleges, accredited_colleges_percent, private_colleges_percent
           - Description: Metrics on education infrastructure, enrollment, test performance, and higher education quality.
           - Example Query: "Find districts with more than 25% private schools and average Class 10 math score above 70%."

           **Category: Vocational Training**
           - Columns: iti_seats_per_lac_youth, iti_vacant_seats_percent, iti_seats_top_3_trades_percent, trainer_vacancies_percent, certified_trainers_percent, pmkvy_enrollment_per_lac, pmkvy_assessment_percent, pmkvy_certification_percent
           - Description: Skill readiness and technical training ecosystem quality indicators.
           - Example Query: "List districts where ITI seat vacancy is above 30% and certified trainers are below 50%."

           **Category: Skills & Occupations**
           - Columns: top_skill_1, top_skill_2, top_skill_3, pmkvy_enrollments_top_3_jobs_percent
           - Description: Emerging skill trends and popular PMKVY job enrollments.
           - Example Query: "Show top 3 emerging skills for youth in Jharkhand districts based on 2024 data."

           **Use this table** for analyses involving youth development, education, employment, skills, MSMEs, or related empowerment indicators at district level.

        2. rainfall_data : This table contains daily average rainfall data (in mm) for various districts and states, along with the reporting agency and date information, can be used for monthly and yearly average data too..
        Instructions: Use this table to analyze rainfall patterns by state, district, date, month, year, or agency. You can filter, aggregate, or compare rainfall data across different regions and time periods.
        Example queries:
        Query: Show the total rainfall in Kerala for February 2025.
        Query: List the average rainfall per day in Kannur district for 2025.
        Query: Get the maximum daily rainfall recorded by NRSC VIC MODEL in Kerala.
        
        Note: If a query is not about youth power or related youth development indicators, return "none_of_these".

        Consider the list above, and respond ONLY with one of the file names from the following list:
        [youthpower_district_level_metrics, rainfall_data, none_of_these]
        DO NOT include any reasoning traces or any other text apart from the file name selected from the above list.
    """)
    selected_file, i_tokens, o_tokens = llm_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def rephrase_for_table(query, schema, context, table_name):
    instructions = f"""
    Given the following table schema: {schema} and the context: {context} for the table {table_name}, rephrase the provided query to make it easier for an SQL agent to pull the right data.

    ** INSTRUCTIONS **
    -- Try to specify values according to the "Suggested Value Settings" in the provided context, AS LONG AS they are not specified by the query itself. When specifying these values, keep the query in mind and try to use generic values such as Combined, *, General, All India, etc.
    REMEMBER that these values are to be specified only if they do not clash with the query.
    -- Do NOT specify month or month_numeric in the output.

    There are two types of query that you should be able to handle. These are specified below, with logical reasoning:
        1. Time-related queries: These are queries where you are asked about a certain quantity over a certain time period.
           - In this case, fix the categorical columns as far as possible and then set date / year limits on the query.
           - Example: Query --> CPI for vegetables in Karnataka, June 2021 to June 2022
                      SQL query --> SELECT * FROM {table_name} WHERE state = 'Karnataka' AND year >= 2021 AND year <= 2022 AND group_name = 'Food and Beverages' AND sub_group_name = 'Vegetables' AND sector = 'Combined' AND data_source = 'Price Statistics Division, MoSPI' LIMIT 125;
           REMEMBER: Do not set month filters in the SQL query, to avoid confusion.

        2. Comparative queries: These are queries that ask for "top k" sort of quantities.
            - For example, this type of query about "top 5 states" which should resolve to top 5 states by GDP, or "top 3 categories" where it should resolve to categories of products.
            - Here, set the time to a specific entry, which should be the latest available in the data.
            - Then, pull all states, categories, etc. that need to be compared without setting any limits on the data.
            - Example: Query --> Top 5 states by GSDP, latest data before June 2025
                    SQL query --> SELECT * FROM {table_name} WHERE year = '2024-25' ORDER BY year DESC
            - Example: Query --> Top 3 categories of inflation, latest data before June 2025
                    SQL query --> SELECT * FROM {table_name} WHERE year = '2025' and month_numeric = '5' AND state = 'All India' AND sector = 'Combined' ORDER BY inflation_rate DESC LIMIT 3;

    ** VERY IMPORTANT: **
    If only one year is specified, DO NOT specify month, month_numeric, or quarter in the year.
    NEVER set the value for data release date or the data source in your query.
    Output your response as a valid SQL query.
        query --> Growth of Maharashtra in the last 10 years
        SQL query --> SELECT * FROM {table_name} WHERE state = 'Maharashtra' AND year >= '2015' AND year <= '2025' AND gross_state_value_added_at = 'current price' AND sector = 'Gross State Domestic Product' AND value_unit = 'lakhs' LIMIT 125;

        query --> Growth in electricity production from June 2020 to June 2022
        SQL query --> SELECT * FROM {table_name} WHERE year >= '2020-21' AND year <= '2022-23' AND sector_type = 'Sectoral' AND category = 'Electricity' AND sub_category = '*' LIMIT 125;
    """
    rephrased, i_tokens, o_tokens = openai_call(instructions, query)
    return rephrased.strip(), i_tokens, o_tokens

def identify_generic_columns(schema):
    try:
        #print("Received schema:" )
        #print(schema)
        lines = ast.literal_eval(schema)
        col_list = []
        for line in lines:
            if "Schema" not in line:
                # Use ast.literal_eval to safely parse the tuple
                # Optionally, convert to dict for clarity
                col_dict = {
                    'column_name': line[0],
                    'data_type': line[1],
                    'nullable': line[2],
                    'default': line[3]
                }
                #print("Found tuple: " + str(col_dict))
                if ("id" not in col_dict['column_name']) and \
                   ("year" not in col_dict['column_name']) and \
                   ("month" not in col_dict['column_name']) and \
                   ("data" not in col_dict['column_name']) and \
                   ("date" not in col_dict['column_name']) and \
                   ("released_on" not in col_dict['column_name']) and \
                   ("updated_on" not in col_dict['column_name']) and \
                   (("character varying" in col_dict['data_type']) or ("text" in col_dict['data_type'])):
                       col_list.append(col_dict['column_name'])
                       #col_list.append("WHERE " + col_dict["column_name"] + " = ")
    except:
        col_list = []
    return col_list

def generate_sql_query(query, schema, context, table_name, last_error="N/A"):
    #if "resulted in no data" not in last_error:
        #columns = str(identify_generic_columns(query, schema))
        #query = str(query) + '''\nEnsure you set the following columns to generic values: ''' + columns)
    if last_error == "N/A":
        sql_query, i_tokens, o_tokens = rephrase_for_table(query, schema, context, table_name)
        return sql_query, query, i_tokens, o_tokens

    instructions = dedent(f"""Given the following table context for {table_name}: {context}\nCan you generate a valid SQL query to get the contents for the natural language query attached below? Be very specific and make sure you output ONLY the SQL query as a string without any other text. Remember to pull all the informative columns in the table, and not just the requested values.
        **Some simple hints to use**
        - Unless the query specifies "data for all categories" or "data for all states", you may choose to pull data for All India, General category, Combined sector, * sub-category, etc. You will get some hints from the sample rows.
        - If the query does specify all categories or a comparison between two categories, states, commodities, etc., then make sure you pull data for all the required fields.
        - Does the provided schema contain any columns called "group", "subgroup", "sector", "state", or similar categorical identifiers? Be sure to set such columns to generic values UNLESS they have been specified in the query.
        - Do the sample rows contain any entries with keywords such as "general", "combined", "All India", or "*"? Using the provided schema, set such values for the relevant columns in the output query, UNLESS the corresponding columns have already been specified. DO NOT specify values for data release dates.
        - When handling time period queries, ensure you pull data for the ENTIRE time period and not for specific months only.
        Example: If the query is "Inflation in food and beverages from May 2023 to May 2025", then use "year >= 2023 AND year >= 2025" and DO NOT set "month_numeric = 5"
        - If you are handling a previous error "SQL query pulled too many rows", then you MUST use the suggested columns and set them to generic values such as "*", "Combined", etc.
        - DO NOT set any values in the SELECT condition. Use your value setting in relevant columns ONLY in the WHERE clause.
        Example:
            ALLOWED: SELECT column1, column2, state, year FROM {table_name} WHERE year = 2025 AND state = 'All India'
            NOT ALLOWED: SELECT column1, column2, year, 'All India' as state FROM {table_name} WHERE year = 2025
        VERY IMPORTANT:
        Do not specify months if the query asks for the whole year. If a date range is specified, make sure you pull ALL the data between those dates.
        If only one year is specified, then DO NOT specify a particular month or quarter.
        EXAMPLE:
            query --> CPI for vegetables in Karnataka, 2021
            SQL query --> SELECT * FROM {table_name} WHERE state = 'Karnataka' AND year = 2021 AND group_name = 'Food and Beverages' AND sub_group_name = 'Vegetables' AND sector = 'Combined' AND data_source = 'Price Statistics Division, MoSPI' LIMIT 125;
        """)
    if last_error != "N/A":
        instructions += dedent(f"""EXTREMELY IMPORTANT: Keep in mind that your last attempt returned the error: {last_error}
        """)
    sql_query, i_tokens, o_tokens = openai_call(instructions, query)
    return sql_query.strip(), query, i_tokens, o_tokens

def table_citation(selected_file):
    table_list = {
    'whole_sale_price_index_wpi_financial_year_wise': "Ministry of commerce and industry",
    'quaterly_estimates_of_expenditure_components_gdp': "MoSPI",
    'provisional_estimateso_gdp_macro_economic_aggregates': "MoSPI",
    'per_capita_income_product_final_consumption': "MoSPI",
    'key_aggregates_of_national_accounts': "MoSPI",
    'quaterly_estimates_of_gdp': "MoSPI",
    'annual_estimate_gdp_growth_rate': "MoSPI",
    'gross_state_value': "MoSPI",
    'cpi_worker_data': "Ministry of Finance",
    'annual_estimate_gdp_crore': "MoSPI",
    'iip_data': "Economic Statistics Division, MoSPI",
    'city_wise_housing_price_indices': "Ministry of Finance",
    'consumer_price_index_cpi_for_agricultural_and_rural_labourers': "MoSPI",
    'whole_sale_price_index_wpi_calendar_wise': "Ministry of commerce and industry",
    'cpi_state_mth_subgrp': "Price Statistics Division, MoSPI",
    'iip_annual_data': "Economic Statistics Division, MoSPI",
    'Iip_india_mth_catg_view': "Economic Statistics Division, MoSPI",
    'iip_india_mth_subcatg_view': "Economic Statistics Division, MoSPI",
    # made the iip changes here
    'iip_in_assam': "Directorate of Economics & Statistics, Government of Assam",
    'cpi_food_worker_data': "Ministry of Finance",
    'msme_state_ureg_recent': "Ministry of MSME",
    # 'msme_sector_growth_rates': "RBI",
    # 'msme_global_data': "Asian Development Bank / ERDI",
    'msme_definitions_by_sector': "Asian Development Bank",
    'msme_priority_sector_view': "Ministry of MSME",
    'msme_industry_view': "Ministry of MSME",
    'msme_share_by_sector_view': "Ministry of MSME",
    'msme_share_by_region_view': "Ministry of MSME",
    'msme_gdp_by_sector_view': "Ministry of MSME",
    'msme_gdp_by_region_view': "Ministry of MSME",
    'msme_gbc_food_non_food_view': "Ministry of MSME",
    'msme_gbc_non_food_dtl_view': "Ministry of MSME",
    'nifty_sme_index_daily_values': "NSE Indices",
    'annual_survey_of_industries': "Ministry of Statistics and Programme Implementation",
    'periodic_labour_force_survey': "National Sample Survey Office, MoSPI",
    'statewise_nsdp': "Ministry of Statistics and Programme Implementation",
    'statewise_nsva': "Ministry of Statistics and Programme Implementation",
    'statewise_pcnsdp': "Ministry of Statistics and Programme Implementation",
    "sa_agri_hhs_crop_sale_quantity_by_agency_major_disposal": "National Sample Survey Office, MoSPI",
    "sa_agri_hhs_reporting_use_of_diff_farming_resources": "National Sample Survey Office, MoSPI",
    "sa_agri_hhs_use_purchased_seed_by_quality": "National Sample Survey Office, MoSPI",
    "sa_avg_expenditure_and_receipts_on_farm_and_nonfarm_assets": "National Sample Survey Office, MoSPI",
    "sa_avg_gross_cropped_area_value_quantity_crop_production": "National Sample Survey Office, MoSPI",
    "sa_avg_monthly_expenses_and_receipts_for_crop_production": "National Sample Survey Office, MoSPI",
    "sa_avg_monthly_total_expenses_crop_production": "National Sample Survey Office, MoSPI",
    "sa_avg_monthly_total_expenses_receipts_animal_farming_30_days": "National Sample Survey Office, MoSPI",
    "sa_dist_agri_hh_not_insuring_crop_by_reason_for_selected_crop": "National Sample Survey Office, MoSPI",
    "sa_dist_agri_hhs_seed_use_by_agency_of_procurement": "National Sample Survey Office, MoSPI",
    "sa_dist_hhs_leasing_out_land_and_avg_area_social_group": "National Sample Survey Office, MoSPI",
    "sa_dist_of_agri_hhs_reporting_use_of_purchased_seed": "National Sample Survey Office, MoSPI",
    "sa_dist_of_hhs_by_hh_classification_for_diff_classes_of_land": "National Sample Survey Office, MoSPI",
    "sa_distribution_hhs_leasing_in_land_avg_area_social_group": "National Sample Survey Office, MoSPI",
    "sa_distribution_loan_outstanding_by_source_of_loan_taken": "National Sample Survey Office, MoSPI",
    "sa_distribution_operational_holdings_by_possession_type": "National Sample Survey Office, MoSPI",
    "sa_est_num_of_hhs_for_each_size_class_of_land_possessed": "National Sample Survey Office, MoSPI",
    "sa_estimated_no_of_hhs_for_different_social_groups": "National Sample Survey Office, MoSPI",
    "sa_no_of_hhs_owning_of_livestock_of_different_types": "National Sample Survey Office, MoSPI",
    "sa_no_per_1000_distri_of_agri_hhs_reporting_sale_of_crops": "National Sample Survey Office, MoSPI",
    "sa_no_per_hh_operational_holding_by_size_hh_oper_holding": "National Sample Survey Office, MoSPI",
    "sa_per_1000_agri_hh_insured_experienced_crop_loss": "National Sample Survey Office, MoSPI",
    "sa_per_1000_crop_producing_hh_crop_disposal_agency_sale_satisf": "National Sample Survey Office, MoSPI",
    "sa_perc_dist_of_land_for_hhs_belonging_operational_holding": "National Sample Survey Office, MoSPI",
    "sa_percent_distribution_of_leased_out_land_by_terms_of_lease": "National Sample Survey Office, MoSPI",
    "mis_access_to_improved_source_of_drinking_water": "National Sample Survey Office, MoSPI",
    "mis_access_to_mass_media_and_broadband": "National Sample Survey Office, MoSPI",
    "mis_availability_of_basic_transport_and_public_facility": "National Sample Survey Office, MoSPI",
    "mis_different_source_of_finance": "National Sample Survey Office, MoSPI",
    "mis_exclusive_access_to_improved_latrine": "National Sample Survey Office, MoSPI",
    "mis_household_assets": "National Sample Survey Office, MoSPI",
    "mis_improved_latrine_and_hand_wash_facility_in_households": "National Sample Survey Office, MoSPI",
    "mis_improved_source_of_drinking_water_within_household": "National Sample Survey Office, MoSPI",
    "mis_income_change_due_to_migration": "National Sample Survey Office, MoSPI",
    "mis_main_reason_for_leaving_last_usual_place_of_residence": "National Sample Survey Office, MoSPI",
    "mis_main_reason_for_migration": "National Sample Survey Office, MoSPI",
    "mis_possession_of_air_conditioner_and_air_cooler": "National Sample Survey Office, MoSPI",
    "mis_usage_of_mobile_phone": "National Sample Survey Office, MoSPI",
    "mis_usual_place_of_residence_different_from_current_place": "National Sample Survey Office, MoSPI",
    "asuse_est_annual_emoluments_per_hired_worker": "Ministry of Statistics and Programme Implementation",
    "asuse_est_annual_gva_per_establishment": "Ministry of Statistics and Programme Implementation",
    "asuse_est_num_establishments_pursuing_mixed_activity": "Ministry of Statistics and Programme Implementation",
    "asuse_est_num_workers_by_employment_gender": "Ministry of Statistics and Programme Implementation",
    "asuse_est_value_key_characteristics_by_workers": "Ministry of Statistics and Programme Implementation",
    "asuse_estimated_annual_gva_per_worker_rupees": "Ministry of Statistics and Programme Implementation",
    "asuse_estimated_number_of_workers_by_type_of_workers": "Ministry of Statistics and Programme Implementation",
    "asuse_per1000_estb_by_hours_worked_per_day": "Ministry of Statistics and Programme Implementation",
    "asuse_per1000_estb_by_months_operated_last_365days": "Ministry of Statistics and Programme Implementation",
    "asuse_per1000_estb_registered_under_acts_authorities": "Ministry of Statistics and Programme Implementation",
    "asuse_per1000_estb_using_computer_internet_last365_days": "Ministry of Statistics and Programme Implementation",
    "asuse_per1000_of_estb_using_internet_by_type_of_its_use": "Ministry of Statistics and Programme Implementation",
    "asuse_per1000_proppartn_estb_by_edu_owner_mjr_partner": "Ministry of Statistics and Programme Implementation",
    "asuse_per1000_proppartn_estb_by_other_econ_activities": "Ministry of Statistics and Programme Implementation",
    "asuse_per1000_proppartn_estb_by_socialgroup_owner": "Ministry of Statistics and Programme Implementation",
    "asuse_per_1000_distri_of_establishments_by_nature_of_operation": "Ministry of Statistics and Programme Implementation",
    "asuse_per_1000_distri_of_establishments_by_type_of_location": "Ministry of Statistics and Programme Implementation",
    "asuse_per_1000_distri_of_establishments_by_type_of_ownership": "Ministry of Statistics and Programme Implementation",
    "asuse_per_1000_of_establishments_which_are_npis_and_non_npis": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_est_num_of_estb_pursuing_mixed_activity": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_est_num_of_estb_serving_as_franchisee_outlet": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_est_num_of_worker_by_employment_and_gender": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_estimated_annual_emoluments_per_hired_worker": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_estimated_annual_gva_per_establishment_rupees": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_estimated_annual_gva_per_worker_rupees": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_estimated_number_of_workers_by_type_of_workers": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_per1000_distri_of_estb_by_nature_of_operation": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_per1000_distri_of_estb_by_type_of_location": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_per1000_distri_of_estb_by_type_of_ownership": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_per1000_estb_by_hours_worked_per_day": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_per1000_estb_by_month_num_operated_last365_day": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_per1000_estb_maintain_post_bank_saving_acc": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_per1000_estb_registered_diff_acts_authorities": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_per1000_estb_use_computer_internet_last365_day": "Ministry of Statistics and Programme Implementation",
    "asuse_statewise_per1000_proppart_estb_by_social_grp_mjr_prtner": "Ministry of Statistics and Programme Implementation",
    "niryat_ite_commodity": "Directorate General of Foreign Trade (DGFT), Ministry of Commerce & Industry",
    "niryat_ite_state": "Directorate General of Foreign Trade (DGFT), Ministry of Commerce & Industry",
    'imf_dm_export': "International Monetary Fund (IMF)",
    'iip_in_andra_pradesh_sector_wise': "Directorate of Economics and Statistics, Government of Andhra Pradesh",
    'iip_in_andra_pradesh_sector_industry_wise': "Directorate of Economics and Statistics, Government of Andhra Pradesh",
    'iip_in_andra_pradesh_use_wise': "Directorate of Economics and Statistics, Government of Andhra Pradesh",
    'iip_in_rajasthan_monthly': "Department of Economics and Statistics, Government of Rajasthan",
    'iip_in_rajasthan_fy_index': "Department of Economics and Statistics, Government of Rajasthan",
    'iip_in_rajasthan_two_digit_index': "Department of Economics and Statistics, Government of Rajasthan",
    'iip_in_kerala_fy_index': "Department of Economics and Statistics (Ecostat), Government of Kerala",
    'iip_in_kerala_monthly': "Department of Economics and Statistics (Ecostat), Government of Kerala",
    'iip_in_kerala_quarterly': "Department of Economics and Statistics (Ecostat), Government of Kerala",
        }

    try:
        citation = table_list[selected_file]
    except:
        citation = "Unknown"
    return citation

def data_description(headers):
    system_instruction=dedent("""You are given the following condensed description of the data pulled from internal insights. Can you create a short description of the data in a paragraph between 20 and 50 words? If any json format data is present, also include a couple of insights from the data.
            """)
    description, i_tokens, o_tokens = openai_call(system_instruction, headers, model="gpt-4o-mini")
    return description, i_tokens, o_tokens

def rationalize_information(result, headers, query):
    if query == "":
        query = "Summarize the provided information, and state that this summary is being provided because the data size was too large to answer the query precisely."
    system_instruction=dedent(f"""
                              You are given the following information about {headers}, in json format:
                                  {result}
                              If you are able to answer the query given below with this information, do so. If not, state that a direct answer is not possible but then summarize the data that is provided.
                              IMPORTANT RULES:
                                  1. DO NOT hallucinate any new information, use only the information provided.
                                  2. DO NOT use your own knowledge, use only the information provided.
                                  3. IGNORE information about ID of the data points and the "base year".
                                  4. If possible, provide the information as a markdown table. This table should be comprehensive based on the provided data. Concentrate on newer data rather than older data. DO NOT miss out on including all information related to the time range in the query.
                                  5. If tabular representation is not possible, provide the information as nicely formatted text (paragraph of around 200 words) or bullet points (approximately 10).
                                  6. Be very brief and focus on answering the provided query. Do not provide decorative information. However, include all data relevant to the time range in {query}.
                              """)
    rationalized_info, i_tokens, o_tokens = openai_call(system_instruction, query, model="gpt-4o-mini")
    return rationalized_info.strip(), i_tokens, o_tokens

def handle_pandas_response(df, query, orig_query, max_rows, nq):
    df.to_csv("debug_dataframe.csv")
    total_i_tokens, total_o_tokens = 0, 0

    df.fillna('', inplace=True)
    headers = ""
    try:
        if len(df) <= 4:
            rationalized_info = df.to_markdown(index=False)
            #headers_text, i, o = data_description("Data context: " + headers)
            #total_i_tokens += i; total_o_tokens += o
            #result = df.to_dict(orient='records')
            headers_text, i, o = data_description(headers + "\nData: " + str(rationalized_info))
            total_i_tokens += i; total_o_tokens += o
            #return result, headers_text.strip(), total_i_tokens, total_o_tokens
            return {"summarized_info": rationalized_info}, headers_text.strip(), total_i_tokens, total_o_tokens
        # Find single-valued columns
        definite_drops = ["id", "data_release_date", "data_updated_date"]
        single_valued_cols = [col for col in df.columns if (col in definite_drops) or ((df[col].nunique(dropna=False) == 1) and (col.lower() != 'year'))]

        # Append their values to the caption
        for col in single_valued_cols:
            val = df[col].iloc[0]
            headers += f"{col}: {val} | "

        # Drop trailing delimiter if needed
        headers = headers.rstrip(" | ")

        # Drop single-valued columns from the dataframe
        df = df.drop(columns=single_valued_cols)

        # Define keywords that indicate temporal association
        temporal_keywords = ['year', 'month', 'quarter', 'date', 'day', 'week', 'period', 'time']

        # Create a regex pattern from the keywords
        pattern = re.compile('|'.join(temporal_keywords), re.IGNORECASE)

        # Find columns with headers matching any of the temporal keywords
        temporal_cols = [col for col in df.columns if pattern.search(col)]
        selected_temporal_cols = {}

        for keyword in temporal_keywords:
            # Filter matching columns for this keyword
            matches = [col for col in temporal_cols if keyword in col.lower() and 'data' not in col.lower()]

            # Prioritize numeric columns among the matches
            numeric_matches = [col for col in matches if pd.api.types.is_numeric_dtype(df[col])]

            if numeric_matches:
                selected_temporal_cols[keyword] = numeric_matches[0]  # Use the first numeric match
            elif matches:
                selected_temporal_cols[keyword] = matches[0]  # Fallback: first non-numeric match

        # Get the selected columns
        cols_to_merge = list(selected_temporal_cols.values())

        if 'date_stamp' in list(df):
            df['date'] = df['date_stamp'].astype(str)
        else:
            # Merge into a 'date' column
            if len(temporal_cols) == 1 and temporal_cols[0] == "year":
                df['date'] = pd.to_datetime(df['year'].str[:4], format='%Y')
            else:
                df['date'] = df[cols_to_merge].astype(str).agg('-'.join, axis=1)
        # Convert 'date' column to datetime (will produce NaT for unparseable rows)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        has_nat = df['date'].isna().any()
        if has_nat:
            df['date'] = df['year'].str.extract(r'^(\d{4})').astype(int)
            df['date'] = pd.to_datetime(df['date'], format='%Y')

        # Sort the DataFrame by the 'date' columnß
        df = df.sort_values(by='date',ascending=False).reset_index(drop=True)
        try:
            check_top_k, i, o = openai_call("""Consider the query given below. Your task is to identify if this is a query that compares or ranks certain quantities, categories, states, etc. according to some value.
                For example:
                - "Top 5 states GDP"
                - "Top 3 categories by inflation"
                - "Best performing sectors in manufacturing"
                - "States with lowest inflation"
                - "States with highest MSME participation"
            ** CLASSIFICATION TASK: YES or NO**
            1. If such comparisons or rankings exists, reply with a single word "YES"
            2. If such rankings do not exist, for example "inflation of food category in 2024", "GDP of India in the last 3 years", "IIP of mining sector in the last decade", then reply with a single word "NO"
            3. Do not reply with anything apart from YES or NO
            4. Do not include any thinking traces""", orig_query, model="gpt-4o-mini")
            total_i_tokens += i; total_o_tokens += o
            if check_top_k == "YES":
                latest_date = df.loc[0, 'date']
                df = df[df['date'] == latest_date]
        except Exception as e:
            print("Could not assess whether query should keep latest date only: " + str(e))

        df = df.drop(columns=["date"])
        nrows = len(df)

        if True:
            rationalized_info = df.to_markdown(index=False)
            headers_text, i, o = data_description("Data context: " + headers)
            total_i_tokens += i; total_o_tokens += o
        else:
            if (nrows <= 12) or (nq == 1):
                result = df.to_dict(orient='records')
                headers_text, i, o = data_description(headers + "\nData: " + str(result))
                total_i_tokens += i; total_o_tokens += o
                return result, headers_text.strip(), total_i_tokens, total_o_tokens

            if 12 < nrows < max_rows:
                result = df.to_dict(orient='records')
                rationalized_info, i_rat, o_rat = rationalize_information(result, headers, orig_query + query)
                total_i_tokens += i_rat; total_o_tokens += o_rat

                headers_text, i_desc, o_desc = data_description(headers)
                total_i_tokens += i_desc; total_o_tokens += o_desc
                return {"summarized_info": rationalized_info.strip()}, headers_text.strip(), total_i_tokens, total_o_tokens

            result = df.to_dict(orient='records')
            rationalized_info, i_rat, o_rat = rationalize_information(result, headers, "Too many rows...")
            total_i_tokens += i_rat; total_o_tokens += o_rat

            headers_text, i_desc, o_desc = data_description(headers)
            total_i_tokens += i_desc; total_o_tokens += o_desc
        return {"summarized_info": rationalized_info.strip()}, headers_text.strip(), total_i_tokens, total_o_tokens
        #return str(rationalized_info).strip(), headers_text.strip(), total_i_tokens, total_o_tokens

    except:
        if "date" in list(df):
            df = df.drop(columns=["date"])

        headers_text, i, o = data_description(headers)
        total_i_tokens += i; total_o_tokens += o

        if len(df) > 100:
            df = df.iloc[:100,:]
        return {"summarized_info": str(df.to_markdown(index=False)).strip()}, headers_text.strip(), total_i_tokens, total_o_tokens