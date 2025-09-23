from   utils_common import llm_call, openai_call
from   textwrap import dedent
import re
import pandas as pd
import ast

def classify_query(query):
    system_instruction=dedent("""
                You are tasked with classifying the given query into one of the following categories: "CPI", "GDP", "IIP", "MSME", "agriculture_and_rural", "social_migration_and_households", "enterprise_establishment_surveys", or "Out of domain". Use the guidelines below to analyze key entities and determine the best fit. Respond only with the selected category name—do not include reasoning, explanations, or additional text.
                              
                You proceed using the following hints:
                              
                # 1. Analyze the provided query for key entities. 

                ## CPI
                              
                a. Classify queries related to inflation, CPI, price indices, sales, wholesale, consumer, or consumption. This includes datasets on general CPI inflation, agricultural and rural laborer price indices, city-wise housing prices, wholesale price indices (financial year and calendar wise), and worker-specific CPI data.
                
                b. The following files comprise the CPI datasets: [ cpi_state_mth_grp_view, cpi_state_mth_subgrp_view, cpi_india_mth_grp_view,cpi_india_mth_subgrp_view, consumer_price_index_CPI_for_agricultural_and_rural_labourers,city_wise_housing_price_indices,whole_sale_price_index_WPI_financial_year_wise,cpi_worker_data,whole_sale_price_index_WPI_calendar_wise]. 
                Any query that can be answered with these data sets should be classified as "CPI".

                ## GDP
                              
                a. Classify queries on GDP, gross domestic product, GSDP, GVA, NDP, NNI, GNI, capital formation, expenditure components, GST (including taxpayers, returns, state contributions, gross/net tax collections, IGST settlements, registrations, subsidies), per capita values, NSDP, NSVA, PCNSDP, state economic indicators, or macro-level metrics like monetary policy instruments (repo rate, bank rate, MSF, SLR, CRR), government securities yields, call money rates, forward premia, balance of payments, external debt, foreign reserves, FDI, portfolio investments, ECB, trade balance, exports, imports, money supply (M1, M3), bank credit, government borrowings, fiscal deficit, market capitalization, and exchange rates. This covers datasets on annual GDP estimates (crore and growth rates), gross state values, national account aggregates, per capita income and consumption, provisional GDP macro aggregates, quarterly expenditure and GDP estimates, other macro indicators (daily, monthly, quarterly, weekly), top macro indicators (monthly, quarterly, weekly), state-wise NSDP/NSVA/PCNSDP, GSTR filings, tax collections and refunds, and GST registrations.
                ** Notes **
                All queries about "CREDIT" should go to MSME and must not go to GDP
                              
                b. The following files comprise the GDP datasets: [annual_estimate_gdp_crore,annual_estimate_gdp_growth_rate,gross_state_value,key_aggregates_of_national_accounts,per_capita_income_product_final_consumption,provisional_estimateso_gdp_macro_economic_aggregates,quaterly_estimates_of_expenditure_components_gdp,quaterly_estimates_of_gdp , other_macro_economic_indicators_daily_data ,other_macro_economic_indicators_monthly_data,other_macro_economic_indicators_quaterly_data,other_macro_economic_indicators_weekly_data, top_fifty_macro_economic_indicators_monthly_data,top_fifty_macro_economic_indicators_quaterly_data,top_fifty_macro_economic_indicators_weekly_data , statewise_nsdp,statewise_nsva,statewise_pcnsdp ,gstr_three_b ,gstr_one , gross_and_net_tax_collection , gst_settlement_of_igst_to_states , gst_statewise_tax_collection_data, gst_statewise_tax_collection_refund_data, gst_registrations, niryat_ite_commodity, niryat_ite_state]. 
                Any query that can be answered with these data sets should be classified as "GDP".

                ## IIP
                              
                a. Entities such as IIP, industrial output, industrial production, mining, manufacturing, electricity, motor vehicles, or other industries should be classified as "IIP".
                
                b. The following files comprise the IIP datasets: [iip_india_yr_catg_view,iip_india_mth_catg_view,iip_india_yr_subcatg_view,iip_india_mth_subcatg_view,iip_in_assam]. 
                
                Any query that can be answered with these data sets should be classified as "IIP".

                ## MSME
                              
                a. Classify queries on MSME, Micro, Small and Medium Enterprises, credit growth, exchange rates, Nifty SME, food/non-food credit, gross bank credit, regional/sectoral MSME distribution, or economic shares. This covers datasets on gross bank credit for food/non-food views, sector definitions, non-food credit details, regional and sectoral shares, industry views, priority sector views, daily Nifty SME index values, and state-wise Udyam registrations.
                              
                b. The following files comprise the MSME datasets: [msme_gbc_food_non_food_view, msme_definitions_by_sector, msme_state_ureg_recent, msme_gbc_non_food_dtl_view, nifty_sme_index_daily_values , msme_share_by_region_view, msme_share_by_sector_view, msme_priority_sector_view, msme_industry_view, msme_global_view]. 
                Any query that can be answered with these data sets should be classified as "MSME".

                ## agriculture_and_rural
                              
                a. Classify queries seeking descriptive statistics on agricultural households, including crop sales, farming inputs, irrigation, receipts, crop insurance (uptake, reasons for non-insurance), farmer advisory, MSP awareness, input procurement, land leasing/ownership, livestock, household classification, credit sources, asset investment, operational holdings, production/yield, and social/regional comparisons. This aligns with themes like agricultural extension, insurance design, rural finance, food security, market linkages, land reform, asset creation, productivity, equity, and development. Datasets include distributions on crop sales by agency, farming resource use, seed quality/procurement, expenditure/receipts on assets and production, crop insurance experiences, land leasing by social group, operational holdings, livestock ownership, and more.

                b. The following files comprise the "agriculture_and_rural" datasets:[sa_agri_hhs_crop_sale_quantity_by_agency_major_disposal, sa_agri_hhs_reporting_use_of_diff_farming_resources, sa_agri_hhs_use_purchased_seed_by_quality, sa_avg_expenditure_and_receipts_on_farm_and_nonfarm_assets, sa_avg_gross_cropped_area_value_quantity_crop_production, sa_avg_monthly_expenses_and_receipts_for_crop_production, sa_avg_monthly_total_expenses_crop_production, sa_avg_monthly_total_expenses_receipts_animal_farming_30_days, sa_dist_agri_hh_not_insuring_crop_by_reason_for_selected_crop, sa_dist_agri_hhs_seed_use_by_agency_of_procurement, sa_dist_hhs_leasing_out_land_and_avg_area_social_group, sa_dist_of_agri_hhs_reporting_use_of_purchased_seed, sa_dist_of_hhs_by_hh_classification_for_diff_classes_of_land, sa_distribution_hhs_leasing_in_land_avg_area_social_group, sa_distribution_loan_outstanding_by_source_of_loan_taken, sa_distribution_operational_holdings_by_possession_type, sa_est_num_of_hhs_for_each_size_class_of_land_possessed, sa_estimated_no_of_hhs_for_different_social_groups, sa_no_of_hhs_owning_of_livestock_of_different_types, sa_no_per_1000_distri_of_agri_hhs_reporting_sale_of_crops, sa_no_per_hh_operational_holding_by_size_hh_oper_holding, sa_per_1000_agri_hh_insured_experienced_crop_loss, sa_per_1000_crop_producing_hh_crop_disposal_agency_sale_satisf, sa_perc_dist_of_land_for_hhs_belonging_operational_holding, sa_percent_distribution_of_leased_out_land_by_terms_of_lease]
                Any query that can be answered with these data sets should be classified as "agriculture_and_rural".

                ## social_migration_and_households
                              
                a. Classify queries on household socio-economic patterns, migration (reasons, remittances, rural-urban shifts, income changes), finance sources, asset ownership (housing, TV, cooler, AC), state-wise sanitation/water access, digital connectivity (mobile, SIM, broadband, mass media), living standards (pucca housing, transport), public health/services, credit, and equity (rural/urban, caste, gender). Datasets cover access to drinking water, mass media/broadband, transport/public facilities, finance sources, latrine/handwashing, household assets, migration reasons/income changes, air conditioner/cooler possession, mobile usage, and residence changes.
                
                b. The following files comprise the "social_migration_and_households" datasets: [mis_access_to_improved_source_of_drinking_water, mis_access_to_mass_media_and_broadband, mis_availability_of_basic_transport_and_public_facility, mis_different_source_of_finance, 
                    mis_exclusive_access_to_improved_latrine, mis_household_assets, mis_improved_latrine_and_hand_wash_facility_in_households, mis_improved_source_of_drinking_water_within_household,
                    mis_income_change_due_to_migration, mis_main_reason_for_leaving_last_usual_place_of_residence, mis_main_reason_for_migration, mis_possession_of_air_conditioner_and_air_cooler,
                    mis_usage_of_mobile_phone, mis_usual_place_of_residence_different_from_current_place]
                Any query that can be answered with these data sets should be classified as "social_migration_and_households".

                ## enterprise_establishment_surveys
                              
                a. Classify queries on establishment/enterprise characteristics like ownership types, owner social groups/education, gender/employment composition, digital adoption (computers/internet), operation nature (perennial/seasonal), location types, registration status, franchisee/NPI status, financial metrics (GVA per establishment/worker, emoluments, outstanding loans, banking access), and operational details (hours/days worked). This includes ASI (Annual Survey of Industries) on industrial establishments, capital/investment, stocks, financials, employment, production; and PLFS (Periodic Labour Force Survey) on employment, unemployment, labor participation. Datasets encompass emoluments/GVA per worker, worker distributions by type/gender, key characteristics, hours/months operated, registrations, computer/internet use, ownership/operation/location distributions, state-wise estimates, and survey data on industries and labor force.
                              
                b. The following files comprise the "enterprise_establishment_surveys" datasets: [asuse_est_annual_emoluments_per_hired_worker, asuse_est_annual_gva_per_establishment, asuse_est_num_establishments_pursuing_mixed_activity, asuse_est_num_workers_by_employment_gender, asuse_est_value_key_characteristics_by_workers, asuse_estimated_annual_gva_per_worker_rupees, asuse_estimated_number_of_workers_by_type_of_workers, asuse_per1000_estb_by_hours_worked_per_day, asuse_per1000_estb_by_months_operated_last_365days, asuse_per1000_estb_registered_under_acts_authorities, asuse_per1000_estb_using_computer_internet_last365_days, asuse_per1000_of_estb_using_internet_by_type_of_its_use, 
                    asuse_per1000_proppartn_estb_by_edu_owner_mjr_partner, asuse_per1000_proppartn_estb_by_other_econ_activities, asuse_per1000_proppartn_estb_by_socialgroup_owner, asuse_per_1000_distri_of_establishments_by_nature_of_operation, asuse_per_1000_distri_of_establishments_by_type_of_location, asuse_per_1000_distri_of_establishments_by_type_of_ownership, asuse_per_1000_of_establishments_which_are_npis_and_non_npis, asuse_statewise_est_num_of_estb_pursuing_mixed_activity, asuse_statewise_est_num_of_estb_serving_as_franchisee_outlet, asuse_statewise_est_num_of_worker_by_employment_and_gender,
                    asuse_statewise_estimated_annual_emoluments_per_hired_worker, asuse_statewise_estimated_annual_gva_per_establishment_rupees, asuse_statewise_estimated_annual_gva_per_worker_rupees, asuse_statewise_estimated_number_of_workers_by_type_of_workers, asuse_statewise_per1000_distri_of_estb_by_nature_of_operation, asuse_statewise_per1000_distri_of_estb_by_type_of_location, asuse_statewise_per1000_distri_of_estb_by_type_of_ownership, asuse_statewise_per1000_estb_by_hours_worked_per_day, asuse_statewise_per1000_estb_by_month_num_operated_last365_day, asuse_statewise_per1000_estb_maintain_post_bank_saving_acc, 
                    asuse_statewise_per1000_estb_registered_diff_acts_authorities, asuse_statewise_per1000_estb_use_computer_internet_last365_day, asuse_statewise_per1000_proppart_estb_by_social_grp_mjr_prtner, annual_survey_of_industries, periodic_labour_force_survey]
                Any query that can be answered with these data sets should be classified as "enterprise_establishment_surveys".

            # 2. Rules for rejection as "Out of Domain"
                              
            Any query that belongs to none of [CPI, GDP, IIP, MSME, agriculture_and_rural, social_migration_and_households, enterprise_establishment_surveys] should be classified "Out of domain". Examples of out of domain queries include those about general queries about the economy, queries about government policies, queries about upcoming challenges, and queries unrelated to finance. All of these should be marked "Out of domain". 
                              
            # 3. EXTREMELY IMPORTANT: 
                - Based on the above description, respond ONLY with one of the classes from the following list:
                    [CPI, GDP, IIP, MSME, agriculture_and_rural, social_migration_and_households, enterprise_establishment_surveys, Out of domain]
                - DO NOT include any reasoning traces or other text apart from the class selected from the above list.
            """)
    # made the iip changes here
    query_class, i_tokens, o_tokens = llm_call(system_instruction, query)
    return query_class.strip(), i_tokens, o_tokens

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

        26. none_of_these: This should be used when the query is unrelated to agricultural household crop sales, crop-selling agencies, or seasonal crop marketing patterns. For example, queries about GDP, inflation, government policies, crop production volume, weather, or support prices (MSP) fall under this category.

        Consider the list above and respond only with one of the following file names:
        [sa_agri_hhs_crop_sale_quantity_by_agency_major_disposal, sa_agri_hhs_reporting_use_of_diff_farming_resources, sa_agri_hhs_use_purchased_seed_by_quality, sa_avg_expenditure_and_receipts_on_farm_and_nonfarm_assets, sa_avg_gross_cropped_area_value_quantity_crop_production, sa_avg_monthly_expenses_and_receipts_for_crop_production, sa_avg_monthly_total_expenses_crop_production, sa_avg_monthly_total_expenses_receipts_animal_farming_30_days, sa_dist_agri_hh_not_insuring_crop_by_reason_for_selected_crop, sa_dist_agri_hhs_seed_use_by_agency_of_procurement, sa_dist_hhs_leasing_out_land_and_avg_area_social_group, sa_dist_of_agri_hhs_reporting_use_of_purchased_seed, sa_dist_of_hhs_by_hh_classification_for_diff_classes_of_land, sa_distribution_hhs_leasing_in_land_avg_area_social_group, sa_distribution_loan_outstanding_by_source_of_loan_taken, sa_distribution_operational_holdings_by_possession_type, sa_est_num_of_hhs_for_each_size_class_of_land_possessed, sa_estimated_no_of_hhs_for_different_social_groups, sa_no_of_hhs_owning_of_livestock_of_different_types, sa_no_per_1000_distri_of_agri_hhs_reporting_sale_of_crops, sa_no_per_hh_operational_holding_by_size_hh_oper_holding, sa_per_1000_agri_hh_insured_experienced_crop_loss, sa_per_1000_crop_producing_hh_crop_disposal_agency_sale_satisf, sa_perc_dist_of_land_for_hhs_belonging_operational_holding, sa_percent_distribution_of_leased_out_land_by_terms_of_lease, none_of_these]
        Do not include any reasoning, explanation, or other text—only respond with the selected file name from the list above.
    """)
    selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_enterprise_establishment_surveys(query):
    system_instruction=dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        1. asuse_est_annual_emoluments_per_hired_worker: This table provides annual, all-India-level data on estimated annual emoluments and number of hired workers by industry (e.g., Manufacture of Rubber and Plastics Products, Land Transport, Education, Accommodation), sector (Rural/Urban/Combined), establishment type (Own Account, Hired Worker), and formality status (Formal/Informal). Example entries include annual emoluments in 'Education' (Urban, 2022-23: Rs 265,897) and hired workers in 'Wholesale on a Fee or Contract Basis' (Combined, Informal, 2023-24: 153,601).
        Sample queries:
        a. What were the estimated annual emoluments per hired worker for Land Transport in rural areas for 2022-23?
        b. How many hired workers were there in the Manufacture of Tobacco Products industry in All India combined for 2021-22?


        2. asuse_est_annual_gva_per_establishment: This table provides annual all-India estimates of Gross Value Added (GVA) per establishment across years (2021-22 to 2023-24), differentiated by urban, rural, and combined regions, activity categories (e.g., Manufacture of Furniture, Accommodation, Other Retail Trade), and establishment types (Own Account Establishments, Hired Worker Establishments, All). Data examples include GVA for 'Other Services', 'Non-captive Electricity Generation', and 'Education' activities, reported by the National Sample Survey Office (MoSPI).
        Sample queries:
        a. What was the estimated annual GVA per establishment for the Manufacture of Furniture category in urban areas in 2021-22?
        b. How does the GVA per establishment for Hired Worker Establishments in Accommodation compare between rural and urban areas in 2023-24?


        3. asuse_est_num_establishments_pursuing_mixed_activity: This table presents annual, all-India estimates of establishments pursuing mixed activities, disaggregated by sector (urban, rural, combined), establishment type (Own Account, Hired Worker, All), and economic activity (e.g., Tobacco Products, Professional Services, Real Estate, Food Manufacturing). Data come from the National Sample Survey Office, MoSPI. Examples: 452,291 urban OAEs in tobacco manufacture (2022-23), 10,336,002 rural retail trade establishments (2023-24), and 24,643,235 all-India service providers (2022-23).
        Sample queries:
        a. How many rural own account establishments were engaged in real estate activities in 2022-23?
        b. What is the estimated number of establishments involved in 'Other Retail Trade' at all-India level for 2023-24?


        4. asuse_est_num_workers_by_employment_gender: This table provides annual, all-India estimates of the number of workers by economic activity (e.g., Manufacturing of Motor Vehicles, Real Estate, Land Transport), nature of establishment (Own Account, Hired Worker, or All), sector (Rural, Urban, Combined), and gender (Male, Female, All). Data is published by the National Sample Survey Office (MoSPI). Sample entries include 'Manufacture of Beverages' (2023-24) and 'Other Retail Trade' (2022-23).
        Sample queries:
        a. How many female workers were engaged in land transport in urban areas during 2021-22?
        b. What was the estimated number of workers in 'Manufacture of Beverages' at all-India level for 2023-24?


        5. asuse_est_value_key_characteristics_by_workers: This table provides annual, All-India level statistics from the National Sample Survey Office (MoSPI) on key economic characteristics by number of workers, broken down by sector (e.g., Manufacturing, Trade, Other Services) and area (Rural, Urban, Combined). Key indicators include output, input, GVA, fixed assets, and outstanding loans per worker and per establishment. For example, in 2023-24, 'GVA per Worker' in rural areas across all categories was 350 (in Rs.'000).
        Sample queries:
        a. What was the output per worker in the manufacturing sector for urban India in 2021-22?
        b. How did fixed assets per worker in the trade sector differ between rural and urban areas in 2023-24?


        6. asuse_estimated_annual_gva_per_worker_rupees: This table presents annual estimates of Gross Value Added (GVA) per worker (in Rs.) across various economic activities in India, categorized by rural, urban, or combined geography. Coverage is at the All-India level, spanning years like 2021-22 to 2023-24. Sectors include manufacturing (e.g., textiles, pharmaceuticals), financial services, education, and trade, and further breaks data into establishment types (Own Account, Hired Worker, All Establishments).
        Sample queries:
        a. What was the estimated annual GVA per worker for the Manufacture of Electrical Equipment in 2021-22 for All Establishments?
        b. How did the GVA per worker in urban areas for Information and Communication activities change from 2022-23 to 2023-24?


        7. asuse_estimated_number_of_workers_by_type_of_workers: This table presents annual, all-India estimates of worker numbers by industry category (e.g., Water Transport, Manufacture of Textiles), worker type (e.g., Formal Hired Workers, Unpaid Family Member), gender, and establishment type (All, Hired Worker, Own Account). Data is disaggregated for urban, rural, and combined areas. Examples include 116 male informal water transport workers (urban, 2023-24) and 2,006 female unpaid family workers in manufacturing (rural, 2022-23).
        Sample queries:
        a. How many informal hired workers were there in urban food and accommodation service activities in 2023-24?
        b. What is the estimated number of female working owners in rural trading activities for 2021-22?


        8. asuse_per1000_estb_by_hours_worked_per_day: This table reports annual, All-India level data on the per-1000 distribution of establishments by hours worked per day, categorized by rural/urban/combined sectors and establishment types such as Own Account Establishments and Hired Worker Establishments. Industry categories include 'Manufacture of Beverages', 'Trading Activities', 'Financial Services', and 'Food and Accommodation Service Activities', with working hours grouped as '<4', '4-7', '8-11', '>11', and 'All'. Example: 705 establishments (urban, professional activities) work '8-11' hours.
        Sample queries:
        a. What percentage of rural establishments in the Manufacture of Beverages category worked more than 11 hours a day in 2023-24?
        b. How does the distribution of working hours differ between urban and rural establishments in the Financial Service Activities Except Insurance and Pension Funding sector in 2022-23?


        9. asuse_per1000_estb_by_months_operated_last_365days: This table presents annual all-India data on the distribution of establishments by months of operation, disaggregated by sector (e.g., Manufacture of Electrical Equipment, Real Estate Activities, Water Transport), type (Own Account, Hired Worker), and rural/urban/combined geographies. Key metrics include per-1000 distribution or average months operated for categories such as '<= 3 Months', '7 to 9 Months', and '> 9 Months'. Data is sourced from the National Sample Survey Office (MoSPI).
        Sample queries:
        a. What was the average number of months operated by rural establishments engaged in the manufacture of leather and related products in 2023-24?
        b. How many per 1000 urban hired worker establishments in wholesale on a fee or contract basis operated for less than or equal to 3 months in 2021-22?


        10. asuse_per1000_estb_registered_under_acts_authorities: This table provides annual, all-India data on the number per 1000 of establishments registered under various Acts or authorities (e.g., Societies Reg. Act, CGST Act, EPFO/ESIC, RTO). The data is category-wise (urban, rural, combined), sector-wise (e.g., Manufacture of Textiles, Information and Communication, Trading Activities), and by type of establishment (All/Own Account/Hired Worker). For example, in 2021-22, 277 per 1000 urban hired worker establishments in wood manufacturing were registered under Shops & Establishments Act.
        Sample queries:
        a. What was the number per 1000 of rural establishments registered under the CGST Act for 'Wholesale and Retail Trade of Motor Vehicles and Motor Cycles' in 2021-22?
        b. How did registration of hired worker establishments in 'Non-captive Electricity Generation and Transmission' change between 2022-23 and 2023-24 under 'Others' in urban areas?


        11. asuse_per1000_estb_using_computer_internet_last365_days: This table presents annual, all-India data from the National Sample Survey Office on the number per 1000 establishments using computers and internet during the last 365 days. The data is available by sector (Trade, Manufacturing, Other Services), location (Rural, Urban, Combined), and type (Own Account Establishments, Hired Worker Establishments, All). For example, in 2023-24, 129 per 1000 urban service establishments used computers, while 640 per 1000 hired worker trade establishments used internet.
        Sample queries:
        a. What was the number per 1000 manufacturing establishments using the internet in urban areas in 2023-24?
        b. How many per 1000 own account trade establishments in rural areas used computers in 2022-23?


        12. asuse_per1000_of_estb_using_internet_by_type_of_its_use: This table provides annual, all-India statistics on the number per 1,000 of establishments using the Internet, broken down by sector (e.g., Manufacturing, Trade, Other Services), area (Rural, Urban, Combined), and specific types of Internet use (such as Internet Banking, Delivering Products Online, Telephoning Over VoIP, Customer Services). Example entries include: Manufacturing establishments in rural India using the Internet for government information in 2022-23 (46), or Urban Trade sector using it for staff training in 2021-22 (323).
        Sample queries:
        a. How many establishments per 1,000 in the Trade sector used Internet banking in urban India in 2021-22?
        b. What was the number per 1,000 of rural manufacturing establishments using the Internet for accessing financial services in 2023-24?


        13. asuse_per1000_proppartn_estb_by_edu_owner_mjr_partner: This dataset provides the annual (financial year) per 1000 distribution of proprietary and partnership establishments in India, categorized by the level of general education of the owner or major partner. Data is available at the all-India level, disaggregated by rural, urban, and combined sectors. Education categories include Not Literate, Literate Below Primary, Literate Graduate and Above, among others. Example: 199 urban establishments per 1000 had graduate owners in 2022-23.
        Sample queries:
        a. What percentage of rural proprietary establishments in 2022-23 were owned by people with primary to below secondary education?
        b. How did the distribution of urban establishments owned by graduates change between 2021-22 and 2023-24?


        14. asuse_per1000_proppartn_estb_by_other_econ_activities: This table provides annual, all-India level data on the distribution per 1,000 Proprietary and Partnership establishments by various economic activities (e.g., Manufacturing Activities, Food and Accommodation Service Activities, Real Estate, Education). Data is disaggregated by rural/urban/combined sectors, establishment size (e.g., Hired Worker Establishments, Own Account Establishments), and the number of other economic activities present. Data spans years like 2021-22 to 2023-24. Source: National Sample Survey Office, MoSPI.
        Sample queries:
        a. What was the per 1000 distribution of proprietary and partnership establishments for Manufacture of Pharmaceuticals in urban India in 2022-23?
        b. Show the annual trend from 2021-22 to 2023-24 for All Establishments in Food and Accommodation Service Activities at the all-India level.


        15. asuse_per1000_proppartn_estb_by_socialgroup_owner: This table presents all-India level, annual data on the per 1000 distribution of proprietary and partnership establishments by the social group of owner or major partner. Data is provided across urban, rural, and combined sectors, for various establishment types (Own Account, Hired Worker, All), and industries such as Manufacture of Rubber and Plastics Products, Accommodation, and Trading Activities. Social group categories include Scheduled Tribe, Scheduled Caste, OBC, Others, and Not Known.
        Sample queries:
        a. What is the distribution of hired worker establishments owned by Scheduled Tribe groups in the financial service sector for 2022-23?
        b. How do the per 1000 establishment distributions for the manufacture of paper and paper products differ between urban and rural areas for OAE in 2023-24?


        16. asuse_per_1000_distri_of_establishments_by_nature_of_operation: This table presents annual all-India estimates of the per 1000 distribution of establishments by nature of operation, disaggregated by sector (e.g., manufacture of food products, education, retail trade), type of establishment (Own Account, Hired Worker, All), and operational status (Perennial, Seasonal, Casual), for rural, urban, and combined regions. Data examples include 'Manufacture of Furniture' (urban, all), 'Other Wholesale Trade' (rural, HWE, seasonal), and 'Human Health and Social Work Activities'.
        Sample queries:
        a. What percentage of urban establishments in India engaged in manufacture of furniture are perennial according to the latest available year?
        b. How does the distribution of hired worker establishments in the education sector vary between rural and urban areas for 2022-23?


        17. asuse_per_1000_distri_of_establishments_by_type_of_location: This table presents annual all-India data on the per 1000 distribution of establishments by type of location, categorized by rural, urban, and combined regions. Industries covered range from 'Other Manufacturing', 'Land Transport', 'Education', 'Financial Service Activities' to 'Manufacture of Chemicals'. Data is further detailed for own account, hired worker, and all establishments across location types such as household premises, permanent and temporary structures, and mobile markets.
        Sample queries:
        a. What is the distribution of 'Land Transport' establishments located outside household premises with permanent structure in rural areas for 2021-22?
        b. How many 'Manufacture of Chemicals and Chemical Products' own account establishments operated within household premises in urban India in 2023-24 per 1000 units?


        18. asuse_per_1000_distri_of_establishments_by_type_of_ownership: This table presents all-India annual data on the per-1000 distribution of establishments by ownership type, for different activity categories (e.g., Manufacture of Textiles, Trading Activities, Accommodation) and sectors (urban, rural, combined). Ownership types include Proprietary-Male, SHG, Partnership, Co-operatives, among others. For example, in 2023-24, 908 per 1000 urban 'Other Financial Activities' establishments were 'Proprietary-Male', and 1000 per 1000 urban 'Accommodation' HWE were 'All'.
        Sample queries:
        a. What percentage of urban establishments under 'Manufacture of Tobacco Products' are owned by women in 2023-24?
        b. How does the distribution of ownership type in 'Accommodation' activities differ between rural and urban India for 2022-23?


        19. asuse_per_1000_of_establishments_which_are_npis_and_non_npis: This table presents annual state-wise data on the number per 1000 of establishments classified as NPIs (Non-Profit Institutions) and non-NPIs, disaggregated by urban, rural, and combined sectors across India. Categories include 'Trade', 'Manufacturing', and 'Other Services', with receipt sources such as 'Donation/Grants' and 'Other Sources'. Examples include 947 non-NPIs per 1000 establishments in urban Nagaland (2023-24) and all-India combined data for manufacturing in 2022-23.
        Sample queries:
        a. What percentage of trade establishments were NPIs with major receipts from donations in Rajasthan (urban) in 2021-22?
        b. Show state-wise data for non-NPI establishments in other services for the year 2022-23.


        20. asuse_statewise_est_num_of_estb_pursuing_mixed_activity: This table provides annual state/UT-wise data on the estimated number of establishments pursuing mixed activities in India, disaggregated by sector (e.g., Manufacturing, Trade, Other Services), area (Urban, Rural, Combined), and establishment type (Own Account Establishments, Hired Worker Establishments, All). Data examples include 763,860 rural manufacturing establishments in Maharashtra (2022-23), 9,483 urban HWE in Chandigarh (2022-23), and 2,197,497 combined OAE in Odisha (2021-22).
        Sample queries:
        a. How many urban hired worker establishments pursuing mixed activities were there in Uttarakhand in 2023-24?
        b. Provide the number of manufacturing establishments in Tamil Nadu (all establishment types) for 2022-23, broken down by area.


        21. asuse_statewise_est_num_of_estb_serving_as_franchisee_outlet: This table contains annual state/UT-wise estimates of the number of establishments serving as franchisee outlets across India from 2021-22 to 2023-24, with data provided at both state (e.g., Bihar: 1129; Odisha: 5826; Telangana: 7308) and all-India levels (e.g., 132471 in 2023-24). Data covers all major states and union territories, including minor entries (e.g., Lakshadweep: 0), and is sourced from the National Sample Survey Office, MoSPI.
        Sample queries:
        a. How many franchisee outlets were estimated in Tamil Nadu in each of the last three years?
        b. Which Indian state had the highest number of franchisee establishments in 2022-23 according to the NSSO?


        22. asuse_statewise_est_num_of_worker_by_employment_and_gender: This table provides annual state/UT-wise estimates of the number of workers in India by industry (e.g., Manufacturing, Trade, Other Services), establishment type (Own Account, Hired Worker, All), gender, work status (Full/Part time), and location (Rural/Urban/Combined). Data covers multiple states such as Assam, Maharashtra, Kerala, and union territories like Chandigarh, for years like 2021-22 to 2023-24. Example: 182,703 male full-time manufacturing workers in urban Andhra Pradesh (2023-24).
        Sample queries:
        a. How many female hired workers were employed full-time in manufacturing establishments in Jammu and Kashmir for 2022-23?
        b. What is the estimated number of part-time male workers in trade sector own account establishments in rural West Bengal for 2023-24?


        23. asuse_statewise_estimated_annual_emoluments_per_hired_worker: This table presents annual, state/UT-wise data on estimated annual emoluments (in Rs.) and hired worker counts across India, sourced from the National Sample Survey Office (MoSPI). Data is available by sector (e.g., Manufacturing, Trade, Other Services), establishment type (Own Account, Hired Worker, All), formality (Formal/Informal/All), and area (Rural, Urban, Combined) for years 2021-22 to 2023-24. Sample entries include Uttar Pradesh, Maharashtra, Meghalaya, Telangana, and Delhi.
        Sample queries:
        a. What was the annual emolument per hired worker in the manufacturing sector for rural Karnataka in 2021-22?
        b. How many hired workers were estimated in all establishments of urban Maharashtra in 2021-22?


        24. asuse_statewise_estimated_annual_gva_per_establishment_rupees: This table provides annual, state/UT-wise and national-level data on estimated Gross Value Added (GVA) per establishment (in INR), compiled by the National Sample Survey Office (MoSPI). Categories include Manufacturing, Trade, Other Services, and aggregation types such as Own Account Establishments (OAE), Hired Worker Establishments (HWE), and All Establishments. Data is further split by rural, urban, and combined areas. Example entries: Mizoram 2022-23 (Rural, OAE), Punjab 2022-23 (Rural, Other Services), All India 2022-23 (Rural, Trade).
        Sample queries:
        a. What was the annual GVA per establishment for trade establishments in urban Haryana for 2023-24?
        b. Compare the estimated annual GVA per establishment for 'Other Services' in Assam and Gujarat over the last three available years.


        25. asuse_statewise_estimated_annual_gva_per_worker_rupees: This table contains annual state/UT-wise estimates of Gross Value Added (GVA) per worker (in Rs.) across India, broken down by sector (Trade, Manufacturing, Other Services), type of establishment (Own Account, Hired Worker, All), and location (Rural, Urban, Combined). Data is available for states like Gujarat, West Bengal, Kerala, Mizoram, and all-India. For example, in 2023-24 Mizoram’s urban hired Trade workers had a GVA per worker of Rs. 336,417.
        Sample queries:
        a. What was the annual GVA per worker in the Manufacturing sector for Gujarat in 2021-22, segmented by establishment type?
        b. How did the GVA per worker in Own Account establishments in Rural Kerala change between 2021-22 and 2023-24?


        26. asuse_statewise_estimated_number_of_workers_by_type_of_workers: This table contains state/UT-wise and all-India annual data on the estimated number of workers by type (e.g., formal/informal hired workers, working owners, unpaid family members) in different sectors (Trade, Manufacturing, Other Services). Data is further broken down by urban/rural/combined, establishment type (All Establishments, Hired Worker, Own Account), and gender. Examples include Rajasthan (rural, Other Services, Other Workers) and Gujarat (urban, Trade, Total workers). Source: National Sample Survey Office, MoSPI.
        Sample queries:
        a. What is the number of female informal hired workers in rural Gujarat in the Trade sector for 2022-23?
        b. Show the annual estimated number of working owners in manufacturing establishments in Madhya Pradesh (urban) for the last three years.


        27. asuse_statewise_per1000_distri_of_estb_by_nature_of_operation: This table provides state/UT-wise, rural/urban/combined, and all-India level annual data on the per 1000 distribution of establishments by nature of operation (perennial, seasonal, casual). Categories covered include Manufacturing, Trade, Other Services, and All, with breakdowns for Own Account Establishments (OAE), Hired Worker Establishments (HWE), and all establishments. Example entries: Uttar Pradesh, Gujarat, Sikkim, Puduchery, Maharashtra (2021–2024, all modes and categories).
        Sample queries:
        a. What percentage of manufacturing establishments in rural Uttar Pradesh were perennial in 2021-22?
        b. Show the state-wise distribution of seasonal own account establishments in the trade sector for 2022-23.


        28. asuse_statewise_per1000_distri_of_estb_by_type_of_location: This table presents annual, state/UT-wise data on the per-1000 distribution of establishments in India, categorized by sector (Manufacturing, Trade, Other Services), establishment type (Own Account Establishments, Hired Worker Establishments, All), and location type (e.g., within household premises, street vendors, kiosks). Coverage includes states like Tamil Nadu, Gujarat, West Bengal, and all-India aggregates, with data available separately for rural, urban, and combined geographies from 2021-22 to 2023-24.
        Sample queries:
        a. What is the distribution of manufacturing establishments by location type in rural West Bengal for 2023-24?
        b. How does the per-1000 share of street vendor establishments in urban Gujarat compare between 2021-22 and 2023-24?


        29. asuse_statewise_per1000_distri_of_estb_by_type_of_ownership: This table provides annual, state/UT-wise and all-India data on the per-1000 distribution of establishments by type of ownership, segmented by rural, urban, and combined areas. Categories include Manufacturing, Trade, and Other Services, with detailed breakdowns such as Proprietary-Male, Partnerships, Societies, and SHGs. Examples include Assam (Rural, Trade), Punjab (Urban, Other Services), and all-India rural data for Trade. Source: National Sample Survey Office, MoSPI.
        Sample queries:
        a. What is the share of proprietary-female owned establishments in the 'Other Services' sector in urban Punjab for 2023-24?
        b. Show per 1000 distribution of SHG-owned establishments for all establishment types in rural Kerala for 2022-23.


        30. asuse_statewise_per1000_estb_by_hours_worked_per_day: This table provides annual, state/UT-wise data on the per-1000 distribution of establishments by the number of hours normally worked per day. It covers various states (e.g., Goa, Maharashtra, Gujarat), sectors (Trade, Manufacturing, Other Services), establishment types (Own Account, Hired Worker, All Establishments), and urban/rural status. Examples include 813 Own Account Trade establishments in Goa and 815 Hired Worker Trade establishments in Assam (2023-24).
        Sample queries:
        a. What is the distribution of manufacturing establishments by hours worked in a day for Telangana (urban) in 2023-24?
        b. Compare the proportion of Own Account versus Hired Worker establishments working 8-11 hours in Gujarat during 2022-23.


        31. asuse_statewise_per1000_estb_by_month_num_operated_last365_day: This table presents annual data on the distribution per 1000 of establishments by number of months operated in the last 365 days, covering different states (e.g., Odisha, Karnataka, Delhi), sectors (Trade, Manufacturing, Other Services), and establishment types (Own Account, Hired Worker, All). Data is available at all-India, state, and urban/rural/combined levels with examples like Odisha (Rural, Trade, <=3 Months) and Delhi (Combined, Manufacturing, <=3 Months).
        Sample queries:
        a. How many own account establishments in Haryana traded for <=3 months during 2023-24?
        b. Which state had the highest per 1000 distribution of trade establishments operating more than 9 months in rural areas in 2022-23?


        32. asuse_statewise_per1000_estb_maintain_post_bank_saving_acc: This table presents annual, state/UT-wise and location-wise (urban/rural/combined) data on the number per 1000 establishments maintaining bank or post office savings accounts in India, split by sectors such as Trade, Manufacturing, and Other Services. Data is categorized for Own Account and Hired Worker Establishments, and by account holder type. Examples include Maharashtra-rural (317, HWE), Assam-urban (914, OAE), and Chandigarh-rural (1000, HWE), from 2021-22 to 2023-24.
        Sample queries:
        a. What was the number per 1000 of urban establishments in Karnataka maintaining any bank account in 2021-22?
        b. How does the proportion of rural own account establishments with Post Office Savings Bank accounts in Punjab compare to Haryana in 2023-24?


        33. asuse_statewise_per1000_estb_registered_diff_acts_authorities: This table provides annual, state/UT-wise and all-India statistics on the number per 1000 of establishments registered under various Acts and authorities (e.g., Shops & Establishment Act, EPFO/ESIC, Co-operative Societies Act) by sector (Trade, Manufacturing, Other Services) and type (Own Account, Hired Worker, All). Examples include 0 per 1000 for 'EPFO/ESIC' in Goa Rural (2021-22) and 148 per 1000 for 'Others' in All India Rural (2022-23).
        Sample queries:
        a. What is the number per 1000 of establishments registered under the Shops & Establishment Act in urban Odisha for 2021-22?
        b. Provide state-wise data for 2023-24 on establishments registered under the Co-operative Societies Act, 1912 in the 'Other Services' sector.


        34. asuse_statewise_per1000_estb_use_computer_internet_last365_day: This table presents annual, state/UT-wise data on the number per 1000 establishments using computers and internet in India, across years like 2021-22 and 2023-24. Data is disaggregated by sector (e.g., Trade, Manufacturing, Other Services), type of establishment (Own Account, Hired Worker, All), and area (Urban, Rural, Combined). Examples include Kerala (rural, Hired Worker, computer: 278), Chandigarh (Combined, Hired Worker, internet: 958), and Uttar Pradesh (urban, all, internet: 158).
        Sample queries:
        a. What is the number per 1000 of trade establishments using computers in urban Sikkim for 2023-24?
        b. How many own account establishments in rural Madhya Pradesh used the internet per 1000 in 2022-23?


        35. asuse_statewise_per1000_proppart_estb_by_social_grp_mjr_prtner: This table provides annual state/UT-wise and national-level data on the per 1000 distribution of proprietary and partnership establishments in India by the social group of the owner-major partner. It covers rural, urban, and combined geographies, establishment types (Own Account, Hired Worker, All), sectors like Manufacturing, Trade, and Other Services, and social groups including Scheduled Castes, Scheduled Tribes, Other Backward Classes, and Others. Example entries: Rajasthan, 2021-22, Manufacturing, Scheduled Caste; All India, 2023-24, Manufacturing, OBC.
        Sample queries:
        a. What is the proportion of Scheduled Tribe-owned manufacturing establishments in Meghalaya (rural) in 2023-24?
        b. How did the distribution of Other Backward Classes in trade sector establishments change in Andhra Pradesh from 2021-22 to 2023-24?

        36. annual_survey_of_industries: This table contains Annual Survey of Industries (ASI) data with comprehensive industrial performance metrics for India. 
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

        37. periodic_labour_force_survey: This table contains Periodic Labour Force Survey (PLFS) data - the primary source for employment and unemployment statistics. Contains Labour Force Participation Rate (LFPR), Worker Population Ratio (WPR), Unemployment Rate (UR) by year, state, gender, age group, sector, religion, social group, education levels.
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

        38. none_of_these: for any queries which are unrelated to above files.

        ## Consider the list above, and respond ONLY with one of the file names from the following list:
        [asuse_est_annual_emoluments_per_hired_worker, asuse_est_annual_gva_per_establishment, asuse_est_num_establishments_pursuing_mixed_activity, asuse_est_num_workers_by_employment_gender, asuse_est_value_key_characteristics_by_workers, asuse_estimated_annual_gva_per_worker_rupees, asuse_estimated_number_of_workers_by_type_of_workers, asuse_per1000_estb_by_hours_worked_per_day, asuse_per1000_estb_by_months_operated_last_365days, asuse_per1000_estb_registered_under_acts_authorities, asuse_per1000_estb_using_computer_internet_last365_days, asuse_per1000_of_estb_using_internet_by_type_of_its_use, 
        asuse_per1000_proppartn_estb_by_edu_owner_mjr_partner, asuse_per1000_proppartn_estb_by_other_econ_activities, asuse_per1000_proppartn_estb_by_socialgroup_owner, asuse_per_1000_distri_of_establishments_by_nature_of_operation, asuse_per_1000_distri_of_establishments_by_type_of_location, asuse_per_1000_distri_of_establishments_by_type_of_ownership, asuse_per_1000_of_establishments_which_are_npis_and_non_npis, asuse_statewise_est_num_of_estb_pursuing_mixed_activity, asuse_statewise_est_num_of_estb_serving_as_franchisee_outlet, asuse_statewise_est_num_of_worker_by_employment_and_gender,
        asuse_statewise_estimated_annual_emoluments_per_hired_worker, asuse_statewise_estimated_annual_gva_per_establishment_rupees, asuse_statewise_estimated_annual_gva_per_worker_rupees, asuse_statewise_estimated_number_of_workers_by_type_of_workers, asuse_statewise_per1000_distri_of_estb_by_nature_of_operation, asuse_statewise_per1000_distri_of_estb_by_type_of_location, asuse_statewise_per1000_distri_of_estb_by_type_of_ownership, asuse_statewise_per1000_estb_by_hours_worked_per_day, asuse_statewise_per1000_estb_by_month_num_operated_last365_day, asuse_statewise_per1000_estb_maintain_post_bank_saving_acc, 
        asuse_statewise_per1000_estb_registered_diff_acts_authorities, asuse_statewise_per1000_estb_use_computer_internet_last365_day, asuse_statewise_per1000_proppart_estb_by_social_grp_mjr_prtner, 
        annual_survey_of_industries , periodic_labour_force_survey , none_of_these]
        
        Do not include any reasoning traces or other text apart from the file name selected from the above list.
            """)
    selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

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

                              
        15. none_of_these: for any queries which are unrelated to above tables.
        
        ## Consider the list above, and respond ONLY with one of the file names from the following list:
        [mis_access_to_improved_source_of_drinking_water, mis_access_to_mass_media_and_broadband, mis_availability_of_basic_transport_and_public_facility, mis_different_source_of_finance, 
        mis_exclusive_access_to_improved_latrine, mis_household_assets, mis_improved_latrine_and_hand_wash_facility_in_households, mis_improved_source_of_drinking_water_within_household,
        mis_income_change_due_to_migration, mis_main_reason_for_leaving_last_usual_place_of_residence, mis_main_reason_for_migration, mis_possession_of_air_conditioner_and_air_cooler,
        mis_usage_of_mobile_phone, mis_usual_place_of_residence_different_from_current_place, none_of_these]
        do not include any reasoning traces or other text apart from the file name selected from the above list.
            """)
    selected_file, i_tokens, o_tokens = openai_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens

def file_selector_CPI(query):
    system_instruction=dedent(f"""
        You are tasked with identifying the file that contains the required data based on the query: "{query}".
        You must pick one file name only from the following list:

        Choose a file only if the table description explicitly confirms that the data required by the query is covered.

        1. cpi_state_mth_grp_view → This view provides monthly Consumer Price Index (CPI) data by state, sector, and group, with inflation indices and rates, filtered for aggregate groups (sub_group_name = '*').
        Columns: base_year, year, month, month_numeric, state, sector, group_name, inflation_index, inflation_rate, released_on, updated_on, data_source, date_stamp, fiscal_year
        Instructions: Use this view to analyze or retrieve state-wise, sector-wise, and group-wise monthly CPI and inflation rates for different years and months. It is suitable for queries requiring aggregate CPI data at the group level (not sub-group).
        Examples:
        User: Show the inflation index for Karnataka (Rural) for Housing in May 2021.
        SQL: SELECT inflation_index FROM cpi_state_mth_grp_view WHERE state = 'Karnataka' AND sector = 'Rural' AND group_name = 'Housing' AND year = 2021 AND month = 'May';

        User: List all available states and sectors in the view.
        SQL: SELECT DISTINCT state, sector FROM cpi_state_mth_grp_view;

        User: Get the inflation rate for Sikkim in July 2022 for the Housing group.
        SQL: SELECT inflation_rate FROM cpi_state_mth_grp_view WHERE state = 'Sikkim' AND year = 2022 AND month = 'July' AND group_name = 'Housing';

        User: Find all records for the fiscal year 2021-22.
        SQL: SELECT * FROM cpi_state_mth_grp_view WHERE fiscal_year = '2021-22';
        
        2. cpi_state_mth_subgrp_view → This table/view contains monthly Consumer Price Index (CPI) data by state, sector, group, and sub-group, including inflation indices and rates, for different years and months in India.
        Columns: id, base_year, year, month, month_numeric, state, sector, group_name, sub_group_name, inflation_index, inflation_rate, released_on, updated_on, data_source, date_stamp, fiscal_year
        Instructions: Use this table/view to retrieve CPI data at the sub-group level for specific states, sectors (Urban/Rural), months, and years. Filter by columns such as state, year, month, sector, group_name, or sub_group_name to get relevant inflation index or rate information.
        Examples:
        User: Show the inflation index for 'Egg' in Bihar (Urban sector) for May 2021.
        SQL: SELECT inflation_index FROM cpi_state_mth_subgrp_view WHERE state = 'Bihar' AND sector = 'Urban' AND year = 2021 AND month = 'May' AND sub_group_name = 'Egg';

        User: List all inflation indices for 'Spices' in Jharkhand (Rural) for fiscal year 2020-21.
        SQL: SELECT year, month, inflation_index FROM cpi_state_mth_subgrp_view WHERE state = 'Jharkhand' AND sector = 'Rural' AND sub_group_name = 'Spices' AND fiscal_year = '2020-21';

        User: Get all CPI data for Madhya Pradesh in May 2021.
        SQL: SELECT * FROM cpi_state_mth_subgrp_view WHERE state = 'Madhya Pradesh' AND year = 2021 AND month = 'May';
        
        3. cpi_india_mth_grp_view → Monthly Consumer Price Index (CPI) data for India at the group level, filtered for 'All India' and all sub-groups.
        Columns: base_year, year, month, month_numeric, state, sector, group_name, inflation_index, inflation_rate, released_on, updated_on, data_source, date_stamp, fiscal_year
        Instructions: Use this view to analyze CPI trends, inflation rates, and index values by year, month, sector, and group for the whole of India. Useful for time series analysis, inflation monitoring, and economic research.
        Examples:
        User: Show the latest CPI inflation rate for the 'General' group in All India.
        SQL: SELECT year, month, inflation_rate FROM cpi_india_mth_grp_view WHERE group_name = 'General' ORDER BY year DESC, month_numeric DESC LIMIT 1;

        User: Get the CPI index and inflation rate for 'Housing' in Urban sector for April 2024.
        SQL: SELECT inflation_index, inflation_rate FROM cpi_india_mth_grp_view WHERE group_name = 'Housing' AND sector = 'Urban' AND year = 2024 AND month = 'April';

        User: List annual average inflation rate for each group in fiscal year 2022-23.
        SQL: SELECT group_name, AVG(inflation_rate::numeric) AS avg_inflation_rate FROM cpi_india_mth_grp_view WHERE fiscal_year = '2022-23' GROUP BY group_name;
        
        4. cpi_india_mth_subgrp_view → This view provides monthly Consumer Price Index (CPI) data for India at the sub-group level, filtered for 'All India' and excluding aggregate sub-groups, with details on inflation index and rate by sector, group, and sub-group.
        Columns: id, base_year, year, month, month_numeric, state, sector, group_name, sub_group_name, inflation_index, inflation_rate, released_on, updated_on, data_source, date_stamp, fiscal_year
        Instructions: Use this view to analyze CPI trends, inflation rates, or index values for specific sub-groups, sectors, or time periods across India. Filter by year, month, sector, group_name, or sub_group_name as needed.
        Examples:
        User: What was the inflation rate for 'Fruits' in November 2021 for the Urban sector?
        SQL: SELECT inflation_rate FROM cpi_india_mth_subgrp_view WHERE year = 2021 AND month = 'November' AND sector = 'Urban' AND sub_group_name = 'Fruits';

        User: Show the inflation index for 'Education' in November 2024 for All India Urban.
        SQL: SELECT inflation_index FROM cpi_india_mth_subgrp_view WHERE year = 2024 AND month = 'November' AND sector = 'Urban' AND sub_group_name = 'Education';

        User: List all sub-groups under 'Food and Beverages' for September 2017 (Combined sector).
        SQL: SELECT sub_group_name FROM cpi_india_mth_subgrp_view WHERE year = 2017 AND month = 'September' AND sector = 'Combined' AND group_name = 'Food and Beverages';

        
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


        11. none_of_these: for any queries which are unrelated to inflation. for example, queries regarding gdp, iip, msme would fall under the "none" category. queries regarding the general state of the economy, government policies, and upcoming challenges also fall under the none_of_these category.
        
        ## Consider the list above, and respond ONLY with one of the file names from the following list:
        [cpi_state_mth_grp_view, cpi_state_mth_subgrp_view, cpi_india_mth_grp_view, cpi_india_mth_subgrp_view, consumer_price_index_cpi_for_agricultural_and_rural_labourers, city_wise_housing_price_indices, whole_sale_price_index_wpi_financial_year_wise, cpi_worker_data, whole_sale_price_index_wpi_calendar_wise, cpi_food_worker_data, none_of_these]
        do not include any reasoning traces or other text apart from the file name selected from the above list.
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

        1. india_fy_gdp_view : Aggregated national-level GDP metrics by financial year. Captures GDP and GVA growth at constant and current prices. Any India level summarization needs for GDP across years, trending and growth rate can be done from this table.
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


        4. gdp_state_fy_actuals → This table contains actual Gross State Domestic Product (GSDP) and related economic indicators for Indian states and union territories by financial year, including values at constant and current prices, per capita figures, year-on-year growth rates, and population estimates.
        Columns: id, state, year, gsva_constant_in_lakhs, gsva_current_in_lakhs, gsdp_constant_in_lakhs, gsdp_current_in_lakhs, population, per_capita_gsdp_constant, per_capita_gsdp_current, gsdp_current_yoy_percent, gsdp_constant_yoy_percent, released_on, updated_on, data_source
        Instructions: Use this table to retrieve state-wise or year-wise GSDP, GSVA, per capita income, population, or growth rates for specific years or states. Filter by 'state', 'year', or other columns as needed to analyze economic performance.
        Examples:
        User: Show the GSDP at current prices for Andaman and Nicobar Islands for 2015-16.
        SQL: SELECT gsdp_current_in_lakhs FROM gdp_state_fy_actuals WHERE state = 'Andaman and Nicobar Islands' AND year = '2015-16';

        User: List all available years and per capita GSDP (constant prices) for Telangana.
        SQL: SELECT year, per_capita_gsdp_constant FROM gdp_state_fy_actuals WHERE state = 'Telangana';

        User: What was the population and GSDP growth rate (constant prices) for Andaman and Nicobar Islands in 2014-15?
        SQL: SELECT population, gsdp_constant_yoy_percent FROM gdp_state_fy_actuals WHERE state = 'Andaman and Nicobar Islands' AND year = '2014-15'; 


        5. gdp_state_fy_industry_actuals_view : This view provides annual GDP data by state and industry, including both constant and current values, with details on data source and release dates.
        Columns: base_year, state, industry, year, constant_value_in_lakh, current_value_in_lakh, released_on, data_source, updated_on
        Instructions: Use this view to retrieve GDP figures for specific states, industries, and years, or to analyze trends in economic output across different sectors and regions.
        Examples:
        User: Show the constant and current GDP values for Bihar in 2011-12 for all industries.
        SQL: SELECT industry, constant_value_in_lakh, current_value_in_lakh FROM gdp_state_fy_industry_actuals_view WHERE state = 'Bihar' AND year = '2011-12';

        User: List all available GDP data for Ladakh in 2022-23.
        SQL: SELECT * FROM gdp_state_fy_industry_actuals_view WHERE state = 'Ladakh' AND year = '2022-23';

        User: Get the manufacturing GDP values for Madhya Pradesh for all years.
        SQL: SELECT year, constant_value_in_lakh, current_value_in_lakh FROM gdp_state_fy_industry_actuals_view WHERE state = 'Madhya Pradesh' AND industry = 'Manufacturing';
        
        
        6. per_capita_income_product_final_consumption : This table contains per capita estimates of key economic indicators in ₹ (Indian Rupees) or growth rate (%) for various years at India Level, along with the population used for those calculations. The indicators relate to income and consumption at both current and constant prices.these measures are ["Per Capita GDP","Per Capita GNI","Per Capita NNI","Per Capita GNDI","Per Capita PFCE" , "Percentage change over previous year at constant (2011-12) prices"]
        Query: "What is the current per capita income in India?"  
        Query: "Give the per capita GDP growth rate for the last five years."  
        Query: "What was the per capita private final consumption expenditure in 2014-15?"  
        Query: "How has per capita GNI changed from 2011-12 to 2024-25?"  
        Query: "Provide the trend of per capita NNI in constant prices over the last decade."  
        Query: "Population used for calculating per capita indicators in 2020-21?"  
        Query: "Show per capita GNDI values and growth rates since 2015."  


        7. gdp_india_fy_estimates_view : This view provides annual GDP estimates for India, including values at constant and current prices, along with their respective growth rates.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze India's GDP figures and growth rates by financial year. Filter by 'year' for specific periods or use 'item' to focus on GDP or other economic indicators.
        Examples:
        User: Show the GDP at constant prices for each year.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_fy_estimates_view WHERE item = 'GDP' ORDER BY year;
        User: List the GDP growth rate at current prices for the last 5 years.
        SQL: SELECT year, growth_rate_current FROM gdp_india_fy_estimates_view WHERE item = 'GDP' ORDER BY year DESC LIMIT 5;
        User: Get all available data for the financial year 2013-14.
        SQL: SELECT * FROM gdp_india_fy_estimates_view WHERE year = '2013-14'

        8.gdp_india_fy_primsector_estimates_view → This view provides annual GDP estimates and growth rates for India's Primary Sector, including both constant and current price values.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze year-wise GDP values and growth rates for the Primary Sector in India, either at constant or current prices.
        Examples:
        User: Show the GDP values at constant prices for the Primary Sector for all available years.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_fy_primsector_estimates_view ORDER BY year;
        User: List the growth rates at current prices for the Primary Sector from 2011-12 to 2013-14.
        SQL: SELECT year, growth_rate_current FROM gdp_india_fy_primsector_estimates_view WHERE year BETWEEN '2011-12' AND '2013-14' ORDER BY year;
        User: Get the most recent data update date for the Primary Sector GDP estimates.
        SQL: SELECT MAX(data_updated_date) AS last_update FROM gdp_india_fy_primsector_estimates_view

        9. top_fifty_macro_economic_indicators_weekly_data: Resolve to this table when the query relates to macroeconomic indicators tracked on a weekly basis for the Indian economy. Focus areas include monetary policy instruments, interest rates (such as repo rate, bank rate, MSF, base rate), yield on government securities and treasury bills, cash reserve ratio (CRR), statutory liquidity ratio (SLR), standing deposit facility (SDF), and forward premia of the US dollar for different durations. Queries about foreign exchange reserves, liquidity conditions, or financial market trends across specific weeks, months, or quarters also belong here. Questions that ask how these economic indicators have changed over time, what the rates were during a particular period, or comparisons between indicators like G-Sec yields and T-bill rates should map to this dataset. Use this table when the intent is to understand the financial health, monetary policy stance, or interest rate environment in India over time.
        Sample queries:
        a. What was the repo rate and reverse repo rate in India on January 27, 2023?
        b. How did the all-India aggregate monetary value change between January 2021 and January 2024?


        10. top_fifty_macro_economic_indicators_quaterly_data:  Resolve to this table when the query includes terms like balance of payments, BoP, overall BoP, net BoP, international investment position, external debt, Indias external debt, gross external debt, or mentions of quarterly external sector data. Use this table if the query asks for the net value of balance of payments, status of India’s international investment position, or total outstanding external debt in US million dollars for a specific quarter, year, or date. 
        Sample queries:
        a. What was India's foreign exchange reserves at the end of Q3 2023?
        b. Show the quarterly trends in India's net capital account balance between 2018 and 2022.


        11. top_fifty_macro_economic_indicators_monthly_data: Resolve to this table when the query involves monthly data on India’s macroeconomic indicators such as commercial paper outstanding, net foreign direct investment (FDI), FDI inflows to India, FDI outflows by India, net portfolio investment, total investment inflows, or foreign currency non-resident (FCNR) bank flows. Also resolve here when the user asks about external commercial borrowings (ECB) registrations, exports, imports, or trade balance in US million dollars. Include queries regarding retail payments, digital payments, and market borrowing by state governments in rupee crore, or monthly exchange rate of the Indian rupee against the US dollar. This table should be used for queries analyzing monthly trends in foreign investment, international trade, capital market activity, currency movement, and payment systems performance.
        Sample queries:
        a. What was India’s foreign exchange reserves in July 2024?
        b. Show the trend in the trade balance for India from 2018 to 2025.


        12. top_fifty_macro_economic_indicators_fortnightly_data: Resolve to this table when the query involves fortnightly data related to India's monetary aggregates, banking sector trends, or investment metrics. Trigger this table for keywords like investment in India, aggregate deposits, cash-deposit ratio, credit-deposit ratio, M3 or broad money, and certificates of deposit outstanding. 
        Sample queries:
        a. What was the repo rate and currency in circulation in India on January 26, 2024?
        b. Show the trend of broad money (M3) in India for the years 2020 to 2023.


        13. other_macro_economic_indicators_weekly_data: Resolve to this table when the query includes keywords related to call money rate, call money borrowing rate, high/low borrowing rates, short-term borrowing rate, or weekly liquidity conditions in the money market. Also trigger this table for queries involving foreign exchange reserves, forex reserves, FX reserves, or foreign currency assets, particularly when values are expressed in crores. Use this table for weekly trends, date-specific values, or comparative analysis of these metrics over short durations. This table is relevant when the query focuses on monetary policy signals, market liquidity, or external sector strength via reserve and rate indicators captured on a weekly basis.
        Sample queries:
        a. What was the interest rate and monetary aggregate on 8 March 2024 at the all-India level?
        b. How have the reported interest rates and monetary aggregates changed from 2015 to 2025?


        14. other_macro_economic_indicators_quaterly_data: Resolve to this table when the query involves quarterly macroeconomic indicators such as Balance of Payments (BoP) components, current account balance, merchandise trade (exports/imports), services trade, net income, transfers, monetary movements, and errors and omissions, in either INR crore or USD million. Trigger this table when keywords like foreign direct investment (FDI), portfolio investment, capital account balance, BoP credit/debit, external loans, or official/private transfers are mentioned. This table supports analysis of India's external sector health, investment flows, real estate trends, and overall economic output on a quarterly frequency. Use this for any in-depth question involving India's external accounts, capital movements.
        Sample queries:
        a. Show the growth in total bank deposits in India from 2010 to 2024.
        b. What was the broad money aggregate value for July 1, 2023, according to the RBI Database?


        15. other_macro_economic_indicators_monthly_data: Resolve to this table for monthly macroeconomic data involving money supply aggregates (M1, M2, M3), domestic credit, bank credit to government or commercial sector, and detailed RBI balance sheet metrics (like assets, liabilities, loans, and currency circulation). Route queries mentioning fiscal deficit, gross primary deficit, government revenue/expenditure, or interest payments here. Use this table for topics related to foreign trade (e.g., exports/imports in ₹ crore), inflation metrics like Wholesale Price Index (WPI) for all commodities or manufactured products, and BSE market capitalization. Additionally, use it when the question concerns monetary aggregates, credit growth, central bank operations, or public currency holdings. This table enables insights into monetary policy effects, fiscal discipline, trade performance, and financial market trends on a monthly frequency.
        Sample queries:
        a. How did total deposits and advances in Indian banks evolve from 1990 to 2020 on a monthly basis?
        b. What was the total amount of monetary aggregates reported in India in May 2023 and how did it compare to May 2018?


        16. other_macro_economic_indicators_daily_data: Resolve to this table when the query involves daily macroeconomic indicators related to India’s financial markets, especially those requiring high-frequency data. Use this table for queries mentioning NSE Nifty, BSE Bankex, repo rate, reverse repo rate, call money rate (high/low), or the RBI’s USD/INR reference rate. It is appropriate when users ask about market reactions, monetary policy effects, or exchange rate changes on specific dates. 
        Sample queries:
        a. What was the USD to INR exchange rate and repo rate on 24 September 2023?
        b. Show Nifty and Sensex values along with bank rates for all available data from 2021.


        17. statewise_nsdp: This table contains statewise Net State Domestic Product (NSDP) data in both constant and current prices with growth rates by state and year. Use for queries about state-level economic performance, NSDP values, and state economic growth.
        Query: "What was the NSDP of Kerala in 2022-23?"
        Query: "Compare Net State Domestic Product growth rates of Tamil Nadu and Karnataka."
        Query: "Show the NSDP trend for Uttar Pradesh over the last 5 years."
        Query: "Which state has the highest NSDP growth rate in 2023-24?"
        Query: "NSDP values at constant prices for Maharashtra from 2018 to 2023."


        18. statewise_nsva: This table contains statewise Net State Value Added (NSVA) data in constant and current prices by state, industry, sub-industry, and economic sectors (primary, secondary, tertiary). Use for queries about sectoral contribution to state economies and industry-wise state performance.
        Query: "Net State Value Added by manufacturing sector in Gujarat."
        Query: "Compare sectoral NSVA between primary, secondary and tertiary sectors in Rajasthan."
        Query: "Agriculture sector contribution to NSVA in Punjab over the last decade."
        Query: "Industry-wise NSVA breakdown for West Bengal in 2022-23."
        Query: "Tertiary sector NSVA growth in Karnataka vs Telangana."


        19. statewise_pcnsdp: This table contains statewise per capita Net State Domestic Product (PCNSDP) data in both constant and current prices with growth rates. Use for queries about per capita income, living standards, and per capita economic performance by state.
        Query: "Per capita NSDP of Goa in 2023-24."
        Query: "Which states have the highest per capita income in India?"
        Query: "Compare per capita NSDP growth rates of Delhi and Mumbai."
        Query: "Show per capita income trends for northeastern states."
        Query: "States with lowest per capita NSDP in the most recent year."


        20. gross_and_net_tax_collection: This dataset provides a detailed breakdown of gross and net tax collections in India, including Goods and Services Tax (GST) revenue. It covers various categories such as Domestic Refunds,  domestic revenue, import revenue, and total GST revenue, Net Revenue Domestic ,Net IGST Revenue ,Export IGST Refunds , Net Revenue ,Export GST Refunds through ICEGATE,further segmented into sub-categories like Central GST (CGST), State GST (SGST), Integrated GST (IGST), and CESS. The data is available on a monthly and yearly basis, sourced from the Reserve Bank of India (RBI). This dataset is crucial for understanding the trends in tax revenue, the impact of GST on different sectors, and the overall fiscal health of the Indian economy. Keywords: Tax collection, GST, CGST, SGST, IGST, CESS, Revenue, India, Monthly, Yearly, Domestic, Imports, Refunds.Do not choose this if state is given in the querty.
        Query: "What was the total gross GST revenue collected in April 2024?",
        Query: "What was the monthly value of CGST collected from domestic sources in April 2023?",
        Query: "What was the total value of refunds related to IGST in April 2024?",
        Query: "What was the gross import revenue collected in April 2023?",
        Query: "What was the net domestic revenue collected in April 2024?"


        21. gstr_one : This dataset provides a comprehensive overview of Goods and Services Tax (GST) return filing behavior across Indian states and union territories. It captures monthly data on the number of taxpayers eligible to file GST returns (GSTR-1), the number who filed by the due date, the number who filed after the due date, the total number of returns filed, and the filing percentage. The data is organized by fiscal year, month, and state, offering a time-series perspective on GST compliance. This dataset is crucial for understanding GST compliance rates, identifying states with high or low compliance, and analyzing trends in filing behavior over time. The source of the data is also provided, along with the dates when the data was released and updated. Keywords: GST, Goods and Services Tax, tax returns, GSTR-1, filing compliance, India, states, fiscal year, monthly data, taxpayers.Also if the query contains "outward supplies", "sales invoices", or "GSTR-1", select gstr_one.
        Query: "What was the total number of GST returns filed in Maharashtra during April 2021-22?",
        Query: "What percentage of eligible taxpayers in Tamil Nadu filed their GST returns by the due date in August 2021-22?",
        Query: "How many taxpayers were eligible to file GST returns in Gujarat during the month of April in the fiscal year 2021-22?",
        Query: "What was the total number of GST returns filed in the state of Kerala during August of the fiscal year 2021-22?",
        Query: "What was the GST filing percentage for the state of Punjab as of June 30th, 2025, for the month of August in the fiscal year 2021-22?"


        22. gstr_three_b : This dataset, provides a comprehensive overview of Goods and Services Tax (GST) return filing compliance across Indian states and union territories. It includes monthly data on the number of taxpayers eligible to file GST returns (GSTR-3B), the number who filed by the due date, the number who filed after the due date, the total number of returns filed, and the filing percentage. The data is categorized by fiscal year, month, and state, offering a granular view of GST compliance trends. This information is sourced from the Reserve Bank of India (RBI) and is updated periodically. The dataset is valuable for understanding state-wise variations in tax compliance, identifying potential areas of tax evasion, and assessing the overall effectiveness of GST implementation. Keywords: GST, tax compliance, India, state-wise data, return filing, GSTR-3B, fiscal year, monthly data, taxpayers, RBI. Also If the query contains "summary return", "tax payment", or "GSTR-3B", select gstr_three_b and if the query only mentions 'GST returns' without specifying, default to gstr_three_b.
        Query: "What was the total number of GST returns filed in Maharashtra during April of the fiscal year 2025-26?",
        Query: "What percentage of eligible taxpayers in Tamil Nadu filed their GST returns by the due date in May of the fiscal year 2025-26?",
        Query: "How many taxpayers were eligible to file GST returns in Gujarat during May of the fiscal year 2025-26?",
        Query: "How many taxpayers in West Bengal filed their GST returns after the due date in April of the fiscal year 2025-26?",
        Query: "What was the total number of GST returns filed in Andhra Pradesh during May of the fiscal year 2025-26?"


        23. gst_settlement_of_igst_to_states : This dataset provides a detailed breakdown of Goods and Services Tax (GST) settlements from the central government to individual states and union territories in India. Specifically, it captures the monthly Integrated GST (IGST) settlements, which include both 'regular' and 'adhoc' components. IGST is levied on inter-state supply of goods and services, and a portion of this revenue is settled with the state where the goods or services are consumed. The data is organized by fiscal year, month, and state, providing a time-series view of these settlements. This dataset is crucial for understanding the fiscal dynamics between the central government and the states, monitoring state revenue streams, and assessing the impact of GST on state finances. Keywords: GST, IGST, settlement, states, fiscal year, month, revenue, India.
        Query: "What was the total GST settlement amount for Maharashtra in the fiscal year 2024-25?",
        Query: "What was the regular GST settlement amount for Tamil Nadu in January of 2026-27?",
        Query: "What was the total GST settlement amount for the state of Gujarat in the month of June in the fiscal year 2023-24?",
        Query: "What was the adhoc GST settlement amount for the state of Kerala in the fiscal year 2022-23?",
        Query: "What was the total GST settlement amount for Punjab in December of the fiscal year 2024-2025?"


        24. gst_statewise_tax_collection_data : This dataset provides a comprehensive record of Goods and Services Tax (GST) collections for each state and union territory in India, broken down by fiscal year and month. It includes the collection amounts for Central GST (CGST), State GST (SGST), Integrated GST (IGST), and CESS. The data is sourced from the Reserve Bank of India (RBI) and includes the dates when the data was released and last updated. This dataset is crucial for understanding the revenue generation of individual states and the overall impact of GST on the Indian economy. It allows for analysis of tax collection trends over time, comparison of performance across different states, and assessment of the contribution of different components of GST to the total revenue. Keywords: GST, tax collection, state revenue, CGST, SGST, IGST, CESS, fiscal year, monthly data, RBI, India.
        Query: "What was the total GST collected in Maharashtra during the fiscal year 2022-23?"
        Query: "What was the SGST collected in Tamil Nadu in the month of December 2021?"
        Query: "What was the CESS collected in Karnataka during the fiscal year 2020-21?"
        Query: "What was the IGST collected in Delhi during the month of June 2022?"
        Query: "What was the total GST collected in India (sum of all states) during the fiscal year 2023-24?"
        Query: "Top 5 GST contributing states in India?"


        25. gst_statewise_tax_collection_refund_data : This dataset provides a detailed, state-wise breakdown of Goods and Services Tax (GST) refunds in India. It includes the amount refunded under various GST components: Central GST (CGST), State GST (SGST), Integrated GST (IGST), and CESS. The data is organized by fiscal year and month, offering a time-series perspective on GST refunds across different states. The data source is specified, typically the Reserve Bank of India (RBI). This dataset is crucial for understanding the flow of GST refunds, assessing the efficiency of the GST system, and analyzing the financial health of individual states. Keywords: GST, Goods and Services Tax, refunds, state-wise, CGST, SGST, IGST, CESS, fiscal year, monthly data, RBI, India, tax refunds, indirect tax.
        Query: "What was the total GST refund amount for Maharashtra in the fiscal year 2022-23?",
        Query: "What was the amount of IGST refund for Tamil Nadu in January 2023?",
        Query: "What was the CESS refund amount for Karnataka in the fiscal year 2021-22?",
        Query: "What was the total refund amount for all states in December 2022?",
        Query: "What was the SGST refund amount for Gujarat in the month of June in the fiscal year 2022-23?"


        26. gst_registrations : This dataset provides a snapshot of Goods and Services Tax (GST) registrations across India, disaggregated by state and taxpayer type. It includes counts of normal taxpayers, composition taxpayers (small businesses with simplified compliance), input service distributors, casual taxpayers, tax collectors at source (TCS), tax deductors at source (TDS), non-resident taxpayers, Online Information Database Access and Retrieval (OIDAR) service providers, and Unique Identity Number (UIN) holders (e.g., embassies). The 'total' column represents the sum of all registration types within each state. This data is crucial for understanding the distribution of businesses registered under GST, assessing the adoption of different GST schemes, and analyzing regional economic activity. Keywords: GST, registrations, taxpayers, state, India, composition scheme, TCS, TDS, OIDAR, UIN.
        Query: "What is the total number of GST registrations in Maharashtra?",
        Query: "How many composition taxpayers are registered in Uttar Pradesh?",
        Query: "What is the number of normal taxpayers registered in Tamil Nadu?",
        Query: "What is the count of tax collectors at source in Gujarat?",
        Query: "How many input service distributors are registered in Karnataka?"


        27. gst_statewise_fiscal_year_collection_view : This table provides annual state-wise data for India on financial metrics (in crore rupees) related to 'RBI Goods and Services', covering multiple years (e.g., 2021-22 to 2025-26). States include Delhi, Assam, Gujarat, Nagaland, Sikkim, and all-India entities like CBIC. Each row details yearly values for different financial categories, the source, and update/publish dates, enabling time-series comparisons across different Indian states.
        “What was the total GST collection of Maharashtra in FY 2022-23?”
        “Which state had the highest GST collection in FY 2021-22?”
        “Show me the GST collection breakdown (CGST, SGST, IGST, Cess) for Tamil Nadu in FY 2020-21.”
        “What is the total GST revenue collected across India in FY 2022-23?”
        “List the top 5 states by GST collection in FY 2019-20.”


        28. gst_statewise_fiscal_year_igst_settlement_view : This table presents annual, state-wise data across Indian states and union territories, detailing figures for variables such as total values, adjustments, and final amounts for 'RBI Goods and Services' (e.g., Gujarat: 27,659; Kerala: 10,939; Lakshadweep: 23). Data spans years like 2020-21 to 2025-26, covers small and large states, and includes all-India territories. Entries feature update dates for each period.
        “How much regular IGST settlement was made to Gujarat in FY 2022-23?”
        “Which state received the highest total IGST settlement in FY 2020-21?”
        “Show me the adhoc IGST settlements for Karnataka in FY 2021-22.”
        “What is the total IGST settlement disbursed across India in FY 2022-23?”
        “Rank the top 3 states by IGST settlement amount in FY 2019-20.”


        29. gst_statewise_fiscal_year_refund_view : This table provides annual, state/UT-wise data on monetary values related to 'RBI Goods and Services' across India. Each row details figures for a financial year, with examples like Gujarat (2023-24), Tamil Nadu (2024-25), and Andaman and Nicobar Islands (2023-24). All-India resolution is included with states and territories as unique entries. Data includes values for different categories over multiple years, last updated in 2025.
        “Give me the SGST refund amount for Karnataka in FY 2022-23.”
        “What is the total refund disbursed across India in FY 2021-22?”
        “Which state received the highest GST refunds in FY 2020-21?”
        “Show me the refund split (CGST, SGST, IGST, Cess) for Maharashtra in FY 2019-20.”
        “List the bottom 5 states by refund amounts in FY 2022-23.”


        30. niryat_ite_commodity : This table provides annual commodity-wise export data for India in million USD, with columns for fiscal year, commodity group, total exports, monthly values (Feb/Mar), month-on-month growth %, share %, and update date. This table provides annual export data for India, disaggregated by product category such as 'Electronic Goods', 'Marine Products', 'Ready-made garments', and 'Drugs And Pharmaceuticals'. Data is available for fiscal years like 2023-24 and 2024-25, showing export value, monthly figures, growth rates, and percentage contribution at the national level. The table includes a 'Total' entry for aggregate exports.",
        “What were India top 5 export commodities in FY 2023-24?”
        “Compare petroleum product and electronic goods exports in the last 3 years.”
        “Show the month-on-month growth in gems and jewellery exports for FY 2024-25.”
        “Which commodity group had the highest export share in FY 2022-23?”

        
        31. niryat_ite_state : This table provides annual state/UT-wise export data for India in million USD, with columns for fiscal year, state/UT, total exports, monthly values (Feb/Mar), month-on-month growth %, share %, and update date. This table provides annual, state-wise financial or production data for Indian states and union territories (e.g., Maharashtra, Kerala, Nagaland, Daman & Diu And Dadra & Nagar Haveli) for financial years such as 2024-25. Each row includes state, year, three quantitative measures, a computed value (possibly percentage change), a ratio or percentage, and a reference date (e.g., 3-Sep-25).
        “Which state exported the most in FY 2022-23?”
        “Rank the top 3 states by export share in FY 2024-25.”
        “Compare Gujarat and Maharashtra exports from FY 2019-20 to FY 2023-24.”
        “What share of India’s exports came from Tamil Nadu in FY 2020-21?”


        32. imf_dm_export : This table contains annual data on a quantitative indicator (e.g., emissions, GDP, or population) for various geographical regions, including individual countries (e.g., Estonia, Brazil, Iran), world regions (South Asia, Europe), and economic groups (Emerging and Developing Asia). The data spans multiple years (e.g., 1982, 2024, 2029). This table provides annual country-wise GDP share (PPP, % of world) from 1980–2029, with columns for country, year, gdp_share_ppp (share of world GDP in PPP terms).
        “What was India’s GDP share in PPP terms in 2022?”
        “List the top 5 countries by share of world GDP in 1980.”
        “How has China’s share of world GDP (PPP) changed from 2000 to 2020?”
        “Which country is projected to have the largest increase in GDP share between 2010 and 2029?”
        “Show the trend of United States vs European Union GDP share from 1980 to 2025.”
        
        33. gdp_india_fy_secsector_estimates_view : This view provides annual GDP estimates for India's secondary sector, including values at constant and current prices, along with their respective growth rates.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze year-wise GDP figures and growth rates for the secondary sector in India, either at constant or current prices.
        Examples:
        User: Show the secondary sector GDP values at constant prices for each year.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_fy_secsector_estimates_view ORDER BY year;
        User: What was the growth rate at current prices for the secondary sector in 2013-14?
        SQL: SELECT growth_rate_current FROM gdp_india_fy_secsector_estimates_view WHERE year = '2013-14';
        User: List all years with their corresponding GDP values and growth rates for the secondary sector.
        SQL: SELECT year, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current FROM gdp_india_fy_secsector_estimates_view ORDER BY year
        
        34. gdp_india_fy_tersector_estimates_view → This view provides annual GDP estimates for India's tertiary sector, including values at constant and current prices, and their respective growth rates.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze year-wise GDP figures and growth rates for the tertiary sector in India, either at constant or current prices.
        Examples:
        User: Show the GDP value and growth rate for the tertiary sector in 2013-14.
        SQL: SELECT year, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current FROM gdp_india_fy_tersector_estimates_view WHERE year = '2013-14';
        User: List all years with their corresponding constant price GDP values for the tertiary sector.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_fy_tersector_estimates_view ORDER BY year;
        User: What was the growth rate at current prices for the tertiary sector in 2012-13?
        SQL: SELECT growth_rate_current FROM gdp_india_fy_tersector_estimates_view WHERE year = '2012-13'
        
        35.gdp_india_fy_primsector_estimates_dtls_view → This table/view provides annual GDP estimates and growth rates for India's primary sector sub-components (like agriculture, livestock, forestry, and fishing), with values at both constant and current prices.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this table/view to retrieve year-wise GDP values and growth rates for specific primary sector items, at constant or current prices, excluding aggregate 'PRIMARY SECTOR' totals.
        Examples:
        User: Show the GDP at constant prices for agriculture, livestock, forestry & fishing for each year.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_fy_primsector_estimates_dtls_view WHERE item = 'AGRICULTURE, LIVESTOCK, FORESTRY & FISHING' ORDER BY year;
        User: List the growth rates at current prices for all primary sector items in 2013-14.
        SQL: SELECT item, growth_rate_current FROM gdp_india_fy_primsector_estimates_dtls_view WHERE year = '2013-14';
        User: Get the most recent data update date for this table.
        SQL: SELECT MAX(data_updated_date) AS last_update FROM gdp_india_fy_primsector_estimates_dtls_view;

        36. gdp_india_fy_secsector_estimates_dtls_view → This table provides annual GDP estimates for India's secondary sector sub-components (excluding the overall 'SECONDARY SECTOR'), including values and growth rates at both constant and current prices.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this table to analyze year-wise GDP values and growth rates for specific secondary sector items (like MANUFACTURING) in India, at both constant and current prices. Filter by 'item' for sub-sector details and by 'year' for time-based analysis.
        Examples:
        User: Show the manufacturing GDP and its growth rate for each year.
        SQL: SELECT year, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current FROM gdp_india_fy_secsector_estimates_dtls_view WHERE item = 'MANUFACTURING' ORDER BY year;
        User: List all secondary sector items with their GDP values for 2013-14.
        SQL: SELECT item, value_in_cr_const, value_in_cr_current FROM gdp_india_fy_secsector_estimates_dtls_view WHERE year = '2013-14';
        User: What was the growth rate at constant prices for each secondary sector item in 2012-13?
        SQL: SELECT item, growth_rate_const FROM gdp_india_fy_secsector_estimates_dtls_view WHERE year = '2012-13';
        
        37. gdp_india_fy_tersector_estimates_dtls_view → This table/view provides annual GDP estimates for various tertiary sector items in India, including values at constant and current prices, along with their growth rates.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this table/view to analyze GDP contributions and growth rates for specific tertiary sector items (excluding the overall 'TERTIARY SECTOR') by year, at both constant and current prices.
        Examples:
        User: Show the GDP value at constant prices for 'TRADE, HOTELS, TRANSPORT, COMMUNICATION & SERVICES RELATED TO BROADCASTING' in 2012-13.
        SQL: SELECT value_in_cr_const FROM gdp_india_fy_tersector_estimates_dtls_view WHERE item = 'TRADE, HOTELS, TRANSPORT, COMMUNICATION & SERVICES RELATED TO BROADCASTING' AND year = '2012-13';
        User: List the growth rates at current prices for all tertiary sector items in 2013-14.
        SQL: SELECT item, growth_rate_current FROM gdp_india_fy_tersector_estimates_dtls_view WHERE year = '2013-14';
        User: Get the GDP values at current and constant prices for each year for 'TRADE, HOTELS, TRANSPORT, COMMUNICATION & SERVICES RELATED TO BROADCASTING'.
        SQL: SELECT year, value_in_cr_const, value_in_cr_current FROM gdp_india_fy_tersector_estimates_dtls_view WHERE item = 'TRADE, HOTELS, TRANSPORT, COMMUNICATION & SERVICES RELATED TO BROADCASTING' ORDER BY year;
        
        38. gdp_india_fy_expenditure_estimates_dtls_view → This view provides annual GDP expenditure estimates for India, including values and growth rates (both constant and current prices) for different expenditure items such as GFCE, with data updated as of the latest available date.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to retrieve GDP expenditure data for India by year and expenditure item, including values and growth rates at both constant and current prices. Filter by 'year', 'item', or use the latest 'data_updated_date' as needed.
        Examples:
        User: Show the constant price GDP expenditure values for all items in 2012-13.
        SQL: SELECT item, value_in_cr_const FROM gdp_india_fy_expenditure_estimates_dtls_view WHERE year = '2012-13';
        User: List the growth rates at current prices for GFCE from 2011-12 to 2013-14.
        SQL: SELECT year, growth_rate_current FROM gdp_india_fy_expenditure_estimates_dtls_view WHERE item = 'GFCE' AND year BETWEEN '2011-12' AND '2013-14';
        User: Get all available data for the year 2013-14.
        SQL: SELECT * FROM gdp_india_fy_expenditure_estimates_dtls_view WHERE year = '2013-14';
        
        39. gdp_india_qtr_estimates_view → This view provides quarterly GDP estimates for India, including values at constant and current prices, along with their respective growth rates, for each year and item where the GDP flag is true.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to retrieve quarterly GDP data for India, including values and growth rates at both constant and current prices, filtered for records marked as GDP.
        Examples:
        User: Show the GDP values at constant prices for the year 2011-12.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_qtr_estimates_view WHERE year = '2011-12' AND item = 'GDP';
        User: Get the growth rates at current prices for all years.
        SQL: SELECT year, growth_rate_current FROM gdp_india_qtr_estimates_view WHERE item = 'GDP';
        User: List all available data update dates in the view.
        SQL: SELECT DISTINCT data_updated_date FROM gdp_india_qtr_estimates_view;
        
        40. gdp_india_qtr_primsector_estimates_view → This view provides quarterly GDP estimates for India's primary sector, including values at constant and current prices, along with their respective growth rates.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze GDP trends, values, and growth rates for the primary sector in India by year and quarter. Filter by 'year' or use the 'data_updated_date' to get the latest data.
        Examples:
        User: Show the constant price GDP values for the primary sector in 2015-16.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_qtr_primsector_estimates_view WHERE year = '2015-16';
        User: Get the growth rates at current prices for the primary sector for all available years.
        SQL: SELECT year, growth_rate_current FROM gdp_india_qtr_primsector_estimates_view;
        User: List all data updated after 01-01-2024.
        SQL: SELECT * FROM gdp_india_qtr_primsector_estimates_view WHERE TO_DATE(data_updated_date, 'DD-MM-YYYY') > '2024-01-01';
        
        41. gdp_india_qtr_secsector_estimates_view → This view provides quarterly GDP estimates for India's secondary sector, including values at constant and current prices, along with their respective growth rates.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze quarterly GDP data specifically for the secondary sector in India, focusing on values and growth rates at both constant and current prices. Filter by year or other columns as needed.
        Examples:
        User: Show the constant price GDP values for the secondary sector for the year 2015-16.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_qtr_secsector_estimates_view WHERE year = '2015-16';
        User: List the growth rates at current prices for each quarter in 2018-19.
        SQL: SELECT year, growth_rate_current FROM gdp_india_qtr_secsector_estimates_view WHERE year = '2018-19';
        User: Get all available data for the secondary sector for the year 2020-21.
        SQL: SELECT * FROM gdp_india_qtr_secsector_estimates_view WHERE year = '2020-21';
        
        42.gdp_india_qtr_tersector_estimates_view → This view provides quarterly GDP estimates for India's tertiary sector, including values at constant and current prices, growth rates, and data update dates.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze GDP figures and growth rates for the tertiary sector in India by year and quarter, with both constant and current price values.
        Examples:
        User: Show the constant price GDP values for the tertiary sector in 2011-12.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_qtr_tersector_estimates_view WHERE year = '2011-12';
        User: List the growth rates at current prices for each year in the tertiary sector.
        SQL: SELECT year, growth_rate_current FROM gdp_india_qtr_tersector_estimates_view;
        User: Get all columns for the latest data update.
        SQL: SELECT * FROM gdp_india_qtr_tersector_estimates_view WHERE data_updated_date = (SELECT MAX(data_updated_date) FROM gdp_india_qtr_tersector_estimates_view);
        
        43. gdp_india_qtr_primsector_estimates_dtls_view → This view provides quarterly GDP estimates for India's primary sector, detailing values and growth rates for specific items like agriculture and mining, at both constant and current prices.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze or compare GDP figures and growth rates for primary sector components (excluding the overall 'PRIMARY SECTOR') by year and item, at constant or current prices.
        Examples:
        User: Show the constant price GDP values for agriculture for each year.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_qtr_primsector_estimates_dtls_view WHERE item = 'AGRICULTURE, LIVESTOCK, FORESTRY & FISHING';

        User: List all items and their growth rates at current prices for 2012-13.
        SQL: SELECT item, growth_rate_current FROM gdp_india_qtr_primsector_estimates_dtls_view WHERE year = '2012-13';

        User: Get the latest updated date for the mining sector data.
        SQL: SELECT MAX(data_updated_date) FROM gdp_india_qtr_primsector_estimates_dtls_view WHERE item = 'MINING & QUARRYING';
        
        44. gdp_india_qtr_secsector_estimates_dtls_view → This view provides quarterly GDP estimates for India's secondary sector, detailing values and growth rates (both constant and current prices) for sub-sectors like manufacturing, construction, and utilities.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze or retrieve GDP data for secondary sector sub-industries in India by year, including their values and growth rates at both constant and current prices. Filter by 'year' or 'item' as needed.
        Examples:
        User: Show the constant price GDP values for manufacturing from 2015-16 onwards.
        SQL: SELECT year, value_in_cr_const FROM gdp_india_qtr_secsector_estimates_dtls_view WHERE item = 'MANUFACTURING' AND year >= '2015-16';

        User: List the growth rates at current prices for all secondary sector items in 2018-19.
        SQL: SELECT item, growth_rate_current FROM gdp_india_qtr_secsector_estimates_dtls_view WHERE year = '2018-19';

        User: Get the latest updated GDP values (current and constant) for construction.
        SQL: SELECT year, value_in_cr_const, value_in_cr_current, data_updated_date FROM gdp_india_qtr_secsector_estimates_dtls_view WHERE item = 'CONSTRUCTION' ORDER BY year DESC LIMIT 1; 

        45.gdp_india_qtr_tersector_estimates_dtls_view → This view provides quarterly GDP estimates for India's tertiary sector, detailing values and growth rates (both constant and current prices) for specific service-related items.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze or retrieve GDP data for individual tertiary sector items (such as financial services, public administration, trade, etc.) by year, including their values and growth rates at both constant and current prices.
        Examples:
        User: Show the constant price GDP values for all tertiary sector items in 2015-16.
        SQL: SELECT item, value_in_cr_const FROM gdp_india_qtr_tersector_estimates_dtls_view WHERE year = '2015-16';

        User: List the growth rates at current prices for 'FINANCIAL, REAL ESTATE & PROFESSIONAL SERVICES' across all years.
        SQL: SELECT year, growth_rate_current FROM gdp_india_qtr_tersector_estimates_dtls_view WHERE item = 'FINANCIAL, REAL ESTATE & PROFESSIONAL SERVICES';

        User: Get all available data for 'TRADE, HOTELS, TRANSPORT, COMMUNICATION & SERVICES RELATED TO BROADCASTING' for the year 2018-19.
        SQL: SELECT * FROM gdp_india_qtr_tersector_estimates_dtls_view WHERE item = 'TRADE, HOTELS, TRANSPORT, COMMUNICATION & SERVICES RELATED TO BROADCASTING' AND year = '2018-19';         
        
        46. gdp_india_qtr_expenditure_estimates_dtls_view → This view provides quarterly GDP expenditure estimates for India, including values and growth rates (both constant and current prices) for various expenditure items by year.
        Columns: year, item, value_in_cr_const, growth_rate_const, value_in_cr_current, growth_rate_current, data_updated_date
        Instructions: Use this view to analyze GDP expenditure components, their values, and growth rates for different years. Filter by 'year' or 'item' to focus on specific periods or expenditure categories.
        Examples:
        User: Show the constant price value and growth rate for 'EXPORTS OF GOODS AND SERVICES' in 2011-12.
        SQL: SELECT value_in_cr_const, growth_rate_const FROM gdp_india_qtr_expenditure_estimates_dtls_view WHERE year = '2011-12' AND item = 'EXPORTS OF GOODS AND SERVICES';

        User: List all items and their current price values for the year 2011-12.
        SQL: SELECT item, value_in_cr_current FROM gdp_india_qtr_expenditure_estimates_dtls_view WHERE year = '2011-12';

        User: Get the growth rates at current prices for all items updated on 16-07-2025.
        SQL: SELECT item, growth_rate_current FROM gdp_india_qtr_expenditure_estimates_dtls_view WHERE data_updated_date = '16-07-2025';
        
        47.gdp_state_fy_subindustry_actuals_view → This view provides annual GDP figures at constant and current prices for each sub-industry within states, including classification flags and product tax/subsidy indicators.
        Columns: id, base_year, state, industry, sub_industry, primary_flag, secondary_flag, tertiary_flag, taxes_on_products, subsidies_on_products, year, constant_value_in_lakh, current_value_in_lakh, released_on, data_source, updated_on
        Instructions: Use this view to retrieve state-wise, industry-wise, and sub-industry-wise GDP data for specific years, including constant and current values, and to filter by economic sector or product tax/subsidy status.
        Examples:
        User: Show the constant and current GDP values for 'Crops' in Andaman Nicobar for all available years.
        SQL: SELECT year, constant_value_in_lakh, current_value_in_lakh FROM gdp_state_fy_subindustry_actuals_view WHERE state = 'Andaman Nicobar' AND sub_industry = 'Crops';

        User: List all sub-industries under 'Agriculture, forestry and fishing' for the year 2013-14 in Andaman Nicobar.
        SQL: SELECT sub_industry FROM gdp_state_fy_subindustry_actuals_view WHERE state = 'Andaman Nicobar' AND industry = 'Agriculture, forestry and fishing' AND year = '2013-14';

        User: Get the GDP values for all primary sector sub-industries in Andaman Nicobar for 2011-12.
        SQL: SELECT sub_industry, constant_value_in_lakh, current_value_in_lakh FROM gdp_state_fy_subindustry_actuals_view WHERE state = 'Andaman Nicobar' AND primary_flag = true AND year = '2011-12';
        
        48. none_of_these: for any queries which are unrelated to GST or GDP. Queries regarding the general state of the economy, government policies, and upcoming challenges also fall under the none_of_these category.

        # Consider the list above, and respond ONLY with one of the file names from the following list:
        [india_fy_gdp_view, india_fy_gdp_components_view, india_fy_national_income_view, gdp_state_fy_actuals, gdp_state_fy_industry_actuals_view, per_capita_income_product_final_consumption, 
        quaterly_estimates_of_gdp, gdp_india_fy_estimates_view, gdp_india_fy_primsector_estimates_view, gdp_india_fy_secsector_estimates_view, gdp_india_fy_tersector_estimates_view,
        other_macro_economic_indicators_daily_data ,other_macro_economic_indicators_monthly_data,other_macro_economic_indicators_quaterly_data,other_macro_economic_indicators_weekly_data,
        top_fifty_macro_economic_indicators_monthly_data,top_fifty_macro_economic_indicators_quaterly_data,top_fifty_macro_economic_indicators_weekly_data, 
        statewise_nsdp, statewise_nsva, statewise_pcnsdp, gst_registrations, gst_statewise_tax_collection_refund_data, gst_statewise_tax_collection_data, gst_settlement_of_igst_to_states, 
        gstr_three_b, gstr_one, gross_and_net_tax_collection, top_fifty_macro_economic_indicators_fortnightly_data, gst_statewise_fiscal_year_collection_view, 
        gst_statewise_fiscal_year_igst_settlement_view, gst_statewise_fiscal_year_refund_view,niryat_ite_commodity,niryat_ite_state,imf_dm_export,gdp_india_fy_primsector_estimates_dtls_view, 
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
        - If the query is BROAD or GENERAL (e.g. just "IIP growth", "IIP trends", "overall industrial performance") — choose the corresponding *category_view* file (monthly or yearly), not the subcategory-specific files.
        - If the query is STATE-SPECIFIC and mentions "Assam" or industries in Assam, then use iip_in_assam.
        - Use the _subcatg_ view if the query mentions a specific, detailed product or industry (e.g., "Manufacture of Food Products," "textiles," "wearing apparel," "motor vehicles"). This view provides the most granular data.
        - Use the _catg_ view for broad, high-level sectors or use-based categories (e.g., General, Mining, Manufacturing, Consumer Durables). Do not use this view if a query mentions a specific product or detailed sub-category.

        # Table information
        1. ip_india_yr_catg_view: This file contains Index of Industrial Production data on an annual or yearly basis at the category level. Use this when queries involve overall IIP or sector/category-level indicators (e.g. General, Manufacturing, Mining, Electricity).
        Sample queries:
        a. What was the annual growth rate of the Mining sector in 2021-22?
        b. Show the General IIP from 2015-16 to 2023-24.
        
        2.iip_india_mth_catg_view: This view contains monthly Index of Industrial Production (IIP) data aggregated at the main category level (e.g., General, Mining, Manufacturing, Consumer Durables). Use this file for: Queries about broad, high-level industrial sectors or use-based categories. Do not use this view if a query mentions a specific product or a detailed sub-category.
        Sample queries:
        a. Show the General index for 2023.
        b. What was the IIP index for Manufacturing sector in August 2023?
        c. Show me the monthly IIP for Primary Goods from January 2023 to June 2023.

        
        3. iip_india_yr_subcatg_view: This file contains Index of Industrial Production data on an annual or yearly basis at the subcategory level. Use this when queries require detailed industry or product breakdowns within categories (e.g. textiles, food products, machinery, chemicals, pharma).
        Sample queries:
        a. Show the IIP growth rate for Manufacture of Textiles in 2023-24.
        b. Which manufacturing subcategories had negative growth in 2022-23?
        
        4. iip_india_mth_subcatg_view: This view contains detailed monthly Index of Industrial Production (IIP) data broken down to the specific sub-category level (e.g., Manufacture of food products, Manufacture of textiles).Use this file for: Queries that ask for data on a specific, detailed industry or product sub-category.
        Sample queries:
        a. List subcategory indices for Manufacture of Food Products in Jan 2025.
        b. Show the IIP growth rate for 'Manufacture of motor vehicles, trailers and semi-trailers' for all of 2023.


        5. iip_in_assam: This file contains Index of Industrial Production (IIP) data specifically for Assam on an annual basis, broken down by industry NIC codes, descriptions, and weights. Use this table if the query is about Assam or state-specific industrial production.
        Sample queries:
        a. What was the Index of Industrial Production for Assam in 2018-19?
        b. Show the IIP for Manufacture of Food Products in Assam from 2012-13 to 2017-18.
        c. Give the industry-wise IIP for Assam in 2016-17.
        d. Trend of IIP in Assam across the last 5 years.


        6. iip_in_andra_pradesh_sector_wise : This table provides the Index of Industrial Production (IIP) for Andhra Pradesh at the sectoral level (e.g., Mining & Quarrying, Manufacturing, Electricity). It is reported monthly and grouped by fiscal year. Use this for broader sector level queries in AP.
        “Show the IIP trend for Andhra Pradesh Mining & Quarrying in 2022–23.”
        “Which sector had the highest index value Andhra Pradesh in July 2023?”
        “Compare the average IIP for Manufacturing Andhra Pradesh between 2021–22 and 2022–23.”
        “What is the IIP of Electricity Andhra Pradesh sector in April 2024?”


        7. iip_in_andra_pradesh_sector_industry_wise : This table provides industry-wise IIP values in Andhra Pradesh, classified by NIC code, industry description, weight, year, and month. It allows analysis at a more granular level within sectors. Use this for specific questions on industries in AP.
        “What was the index value for Manufacture of Textiles in Andhra Pradesh June 2024?”
        “List the top 5 industries by index value in Andhra Pradesh June 2024.”
        “Compare the performance of Manufacture of Food Products vs Manufacture of Beverages in Andhra Pradesh 2024–25.”
        “Which industry showed the largest growth in index value Andhra Pradesh from May to June 2024?”


        8. iip_in_andra_pradesh_use_wise : This table provides the IIP for Andhra Pradesh by use-based classification (e.g., Primary Goods, Capital Goods, Intermediate Goods, Consumer Goods). It is reported monthly across fiscal years. Use this for queries on the consumption side of industries in AP.
        “Show the IIP trend for Primary Goods in Andhra Pradesh 2023–24 vs 2024–25.”
        “What was the Capital Goods index value in Andhra Pradesh April 2024?”
        “Compare the average IIP of Consumer Durables vs Consumer Non-Durables in Andhra Pradesh 2022–23.”
        “Which use-based category Andhra Pradesh had the sharpest decline between April and June 2024?”


        9. iip_in_rajasthan_monthly : This table provides the monthly Index of Industrial Production (IIP) for Rajasthan, categorized into General, Manufacturing, Electricity, and Mining, by fiscal year and month.
        “Show the monthly IIP trend for Manufacturing in Rajasthan 2025–26.”
        “What was the Electricity index in Rajasthan May 2025?”
        “Compare General index values Rajasthan for April, May, and June of 2025–26.”
        “Which category had the lowest index value in Rajasthan June 2025?”


        10. iip_in_rajasthan_fy_index : This table provides the annual average IIP for Rajasthan across categories such as Electricity, Manufacturing, Mining, and the Total General Index, by fiscal year.
        “What was the Total General Index in Rajasthan 2021–22 vs 2022–23?”
        “Which sector showed the highest growth Rajasthan between 2021–22 and 2022–23?”
        “List the average IIP values for Rajasthan Manufacturing across all fiscal years.”
        “Compare the Electricity index Rajasthan between 2021–22 and 2022–23.”


        11. iip_in_rajasthan_two_digit_index : This table provides the IIP for Rajasthan by two-digit NIC industry codes, with industry descriptions and fiscal year index values.
        “What was the IIP value for Manufacture of Textiles in Rajasthan 2025–26?”
        “List the top 5 industries by index value in Rajasthan 2025–26.”
        “Compare Manufacture of Food Products and Manufacture of Beverages in Rajasthan 2025–26.”
        “Which industry had the lowest index value in Rajasthan 2025–26?”

 
        12. iip_in_kerala_fy_index : This table provides the annual Index of Industrial Production (IIP) for Kerala, reported by fiscal year and category (e.g., Manufacturing), along with corresponding index values.
        “What was Kerala’s Manufacturing index in Kerala 2019–20?”
        “Compare the Manufacturing IIP Kerala between 2018–19 and 2020–21.”
        “Show the year-wise trend of Kerala’s Manufacturing IIP from 2015–16 to 2023–24.”
        “In which year did the Manufacturing index reach its lowest value in Kerala?”


        13. iip_in_kerala_monthly : This table provides monthly IIP data for Kerala by fiscal year, category (e.g., General, Manufacturing, Electricity), and month, with corresponding index values.
        “What was the General IIP in Kerala April 2020–21?”
        “Compare April values of General Kerala IIP across fiscal years 2018–19 to 2023–24.”
        “Show the month-on-month trend of General index in Kerala 2019–20.”
        “Which fiscal year had the highest April General index in Kerala?”


        14. iip_in_kerala_quarterly : This table provides quarterly IIP data for Kerala by fiscal year, quarter (Q1, Q2, Q3, Q4), and category (e.g., Manufacturing), with corresponding index values.
        “What was Kerala’s Manufacturing index in Kerala Q2 of 2016–17?”
        “Compare Q1 Manufacturing Kerala values across fiscal years 2015–16 to 2017–18.”
        “Show the quarter-wise Manufacturing Kerala IIP trend for 2016–17.”
        “Which quarter had the highest Manufacturing Kerala index in 2015–16?”

        
        15. none_of_these: for any queries which are unrelated to IIP. Queries regarding the general state of the economy, government policies, and upcoming challenges also fall under the none_of_these category.

        Consider the list above, and respond ONLY with one of the file names from the following list:
        [iip_india_yr_catg_view,Iip_india_mth_catg_view,iip_india_yr_subcatg_view,iip_india_mth_subcatg_view,iip_in_assam,iip_in_andra_pradesh_sector_wise,iip_in_andra_pradesh_sector_industry_wise,iip_in_andra_pradesh_use_wise,iip_in_rajasthan_monthly,iip_in_rajasthan_fy_index,iip_in_rajasthan_two_digit_index,iip_in_kerala_fy_index,iip_in_kerala_monthly,iip_in_kerala_quarterly,none_of_these]
        DO NOT include any reasoning traces or other text apart from the file name selected from the above list.
    """)
    # made the iip changes here
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


        3. msme_state_ureg_recent → This table provides recent state-wise data on MSME (Micro, Small, and Medium Enterprises) registrations, including counts for each category, total registrations, and data source details.
        Columns: state, micro, small, medium, total_udyam, ime_uap, total_msme, data_source, released_on, updated_on
        Instructions: Use this table to retrieve or analyze the latest MSME registration statistics by state, including breakdowns by enterprise size and total counts.
        Examples:
        User: Show the total number of MSMEs registered in each state.
        SQL: SELECT state, total_msme FROM msme_state_ureg_recent;

        User: List the number of micro, small, and medium enterprises in Assam.
        SQL: SELECT micro, small, medium FROM msme_state_ureg_recent WHERE state = 'ASSAM';

        User: Which state has the highest number of total MSMEs?
        SQL: SELECT state, total_msme FROM msme_state_ureg_recent ORDER BY total_msme DESC LIMIT 1;

        User: Get the latest update date for MSME data in Andhra Pradesh.
        SQL: SELECT updated_on FROM msme_state_ureg_recent WHERE state = 'ANDHRA PRADESH';


        4.msme_gbc_non_food_dtl_view : This table contains Monthly Gross Bank Credit (GBC) outstanding by economic sector for non food credit (e.g., Agriculture, Services,Personal Loans,Industries etc.), the column 'gbc_in_cr' holds the amount of Gross Bank Credit (in crores).
        Query1: "What was the Gross Bank Credit to the Agriculture sector in March 2020?"
        Query2: "Show the Personal Loans GBC values across all months of 2021."
        Query3: "How much GBC was given to the Services sector in April 2021?"
        Query4: "List sector-wise GBC on 27 March 2020."
        Query5: "Which sector had the highest GBC in 2021? Provide effective and release dates.


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
        
        11. msme_india_mth_sector_bankcredit_view → Monthly outstanding bank credit data in India by sector, including MSME, personal loans, and agriculture, sourced from the Reserve Bank of India.
        Columns: year, month, sector, effective_date, outstanding_as_on, unit, released_on, updated_on, data_source, date_stamp, month_numeric, fiscal_year
        Instructions: Use this view to analyze monthly outstanding bank credit amounts by sector, year, and month. Filter by 'sector', 'year', or 'month' as needed to get specific credit data. The 'outstanding_as_on' column provides the credit amount (in crore) as of the effective date.
        Examples:
        User: Show the outstanding bank credit for MSME sectors in July 2019.
        SQL: SELECT sector, outstanding_as_on FROM msme_india_mth_sector_bankcredit_view WHERE year = 2019 AND month = 'July' AND sector LIKE '%Micro Small Medium%';
        
        User: List the outstanding credit for all sectors for the fiscal year 2019-20.
        SQL: SELECT sector, month, outstanding_as_on FROM msme_india_mth_sector_bankcredit_view WHERE fiscal_year = '2019-20';

        User: Get the total outstanding bank credit for 'Personal Loans' in 2019.
        SQL: SELECT SUM(outstanding_as_on) AS total_credit FROM msme_india_mth_sector_bankcredit_view WHERE year = 2019 AND sector = 'Personal Loans';
        
        12. msme_india_mth_grp_bankcredit_view → This view provides monthly outstanding bank credit data for different MSME-related sectors and groups in India, sourced from the Reserve Bank of India. It includes details such as year, month, sector, group name, outstanding amount, and relevant dates.
        Columns: year, month, sector, group_name, effective_date, outstanding_as_on, unit, released_on, updated_on, data_source, date_stamp, month_numeric, fiscal_year
        Instructions: Use this view to analyze trends, compare outstanding credit amounts, or extract time-series data for specific MSME sectors or groups by month and year.
        Examples:
        User: Show the total outstanding bank credit for the 'Services' sector in fiscal year 2019-20.
        SQL: SELECT SUM(outstanding_as_on::numeric) AS total_outstanding FROM msme_india_mth_grp_bankcredit_view WHERE sector = 'Services' AND fiscal_year = '2019-20';

        User: List the monthly outstanding amounts for 'Professional Services' group in 2019.
        SQL: SELECT year, month, outstanding_as_on FROM msme_india_mth_grp_bankcredit_view WHERE group_name = 'Professional Services' AND year = 2019 ORDER BY month_numeric;

        User: Get the latest outstanding amount for each sector as of March 2020.
        SQL: SELECT sector, group_name, outstanding_as_on FROM msme_india_mth_grp_bankcredit_view WHERE year = 2020 AND month = 'March';
        
        13. msme_india_mth_subgrp_bankcredit_view → This view provides monthly outstanding bank credit data for various MSME sectors and subgroups in India, including details like sector, group, subgroup, outstanding amount, and reporting dates.
        Columns: year, month, sector, group_name, subgroup, effective_date, outstanding_as_on, unit, released_on, updated_on, data_source, date_stamp, month_numeric, fiscal_year
        Instructions: Use this view to analyze trends in bank credit outstanding amounts for different MSME sectors, groups, and subgroups across months and years. Filter by year, sector, group_name, or subgroup to get specific insights.
        Examples:
        User: Show the outstanding bank credit for 'Power' under 'Infrastructure' group for January 2021.
        SQL: SELECT * FROM msme_india_mth_subgrp_bankcredit_view WHERE group_name = 'Infrastructure' AND subgroup = 'Power' AND year = 2021 AND month = 'January';

        User: List all subgroups and their outstanding amounts for the 'Trade' group in April 2019.
        SQL: SELECT subgroup, outstanding_as_on FROM msme_india_mth_subgrp_bankcredit_view WHERE group_name = 'Trade' AND year = 2019 AND month = 'April';

        User: Get the total outstanding bank credit for 'Industry Micro Small Medium Large' sector in fiscal year 2020-21.
        SQL: SELECT SUM(outstanding_as_on) AS total_outstanding FROM msme_india_mth_subgrp_bankcredit_view WHERE sector = 'Industry Micro Small Medium Large' AND fiscal_year = '2020-21';

        
        Note: If a query is about gdp of msme do not select any table return "none_of_these".

        Consider the list above, and respond ONLY with one of the file names from the following list:
        [msme_gbc_food_non_food_view, msme_definitions_by_sector, msme_state_ureg_recent, msme_gbc_non_food_dtl_view, nifty_sme_index_daily_values , msme_share_by_region_view, msme_share_by_sector_view, 
        msme_priority_sector_view, msme_industry_view, msme_global_view, msme_india_mth_sector_bankcredit_view, msme_india_mth_grp_bankcredit_view, msme_india_mth_subgrp_bankcredit_view, none_of_these]   
        DO NOT include any reasoning traces or other text apart from the file name selected from the above list.
            """)
    selected_file, i_tokens, o_tokens = llm_call(system_instruction, query)
    return selected_file.strip(), i_tokens, o_tokens



#6.msme_gdp_by_region_view : Use this table when the question involves GDP contribution of MSMEs grouped by region, subregion, or country, it contains GDP contribution of MSMEs by region/subregion and country within Asia
#7.msme_gdp_by_sector_view : Use this table when the question involves GDP contribution of MSMEs by sector within a region or country.it contains GDP contribution of MSMEs by sector across Asian countries and regions

### Table Selection Examples, **Few-shot for msme_gdp_by_region_view**
#Select file: msme_gdp_by_region_view for queries like those listed below-
#Query1: "What is the MSME GDP in South Asia in 2022?"
#Query2: "Compare MSME GDP across Southeast Asia and Pacific Islands."

### Table Selection Examples, **Few-shot for msme_gdp_by_region_view**
#Select file: msme_gdp_by_region_view for queries like those listed below-
#Query1: "What is the MSME GDP in South Asia in 2022?"
#Query2: "Compare MSME GDP across Southeast Asia and Pacific Islands."


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
    description, i_tokens, o_tokens = openai_call(system_instruction, headers)
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
    rationalized_info, i_tokens, o_tokens = openai_call(system_instruction, query)
    return rationalized_info.strip(), i_tokens, o_tokens

def handle_pandas_response(df, query, orig_query, max_rows, nq): 
    df.to_csv("debug_dataframe.csv")
    total_i_tokens, total_o_tokens = 0, 0
    
    df.fillna('', inplace=True) 
    headers = ""
    try:
        if len(df) <= 4:
            result = df.to_dict(orient='records')
            headers_text, i, o = data_description(headers + "\nData: " + str(result))
            total_i_tokens += i; total_o_tokens += o
            return result, headers_text.strip(), total_i_tokens, total_o_tokens
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
            4. Do not include any thinking traces""", orig_query)
            total_i_tokens += i; total_o_tokens += o
            if check_top_k == "YES":
                latest_date = df.loc[0, 'date']
                df = df[df['date'] == latest_date]
        except Exception as e:
            print("Could not assess whether query should keep latest date only: " + str(e))
            
        df = df.drop(columns=["date"])
        nrows = len(df)
        
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
    
    except:
        if "date" in list(df):
            df = df.drop(columns=["date"])
        
        headers_text, i, o = data_description(headers)
        total_i_tokens += i; total_o_tokens += o
        
        if len(df) > 100:
            df = df.iloc[:100,:]
        return df.to_dict(orient='records'), headers_text.strip(), total_i_tokens, total_o_tokens
