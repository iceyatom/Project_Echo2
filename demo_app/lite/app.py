"""
Project Echo - Income Predictor (Lite / 3-booster blend)
=========================================================
Interactive tkinter demo over the ACS 2024 PUMS income model (Experiment 4).

This LITE app bundles the three gradient-boosted base learners from the exp4
ensemble -- XGBoost + LightGBM + CatBoost -- and averages them. It is a faithful,
lightweight stand-in for the full 6.7 GB stacking ensemble: the boosters each
score Test R2 ~0.51 and their blend ~0.5125, statistically identical to the full
Stacking(GBMs) model (0.5154), because the wage signal is feature-bound.

Model bundle (model.pkl) is loaded via joblib. Target was trained on log1p(WAGP)
where WAGP is in millions of constant dollars, so predictions are inverted with
expm1 and multiplied by 1e6 to report whole dollars.
"""
import os
import sys
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

warnings.filterwarnings("ignore")   # silence ML-lib chatter so it never reaches the GUI

# --- Locate model.pkl -- works as a plain script and as a PyInstaller .exe -----
def _resolve_model(filenames):
    """Find the model bundle. Looks NEXT TO the .exe / script first (the
    recommended 'drop model.pkl beside the app' layout), then other fallbacks."""
    dirs = []
    if getattr(sys, "frozen", False):
        dirs.append(os.path.dirname(sys.executable))     # beside the .exe  <-- primary
        dirs.append(sys._MEIPASS)                        # type: ignore[attr-defined]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        dirs += [here, os.path.join(here, "..", "..")]   # beside app.py, then repo root
    cands = [os.path.join(d, f) for d in dirs for f in filenames]
    return next((p for p in cands if os.path.exists(p)), cands[0])

MODEL_PATH = _resolve_model(["model.pkl"])


# =============================================================================
# ACS PUMS 2024 option labels  (code -> human-readable label)
# Codes are the raw values the model expects; labels are what the GUI shows.
# =============================================================================

# COW - Class of worker (employment sector)
COW_LABELS = {
    1: "Private, for-profit company",
    2: "Private, non-profit organization",
    3: "Local government employee",
    4: "State government employee",
    5: "Federal government employee",
    6: "Self-employed (own business, not incorporated)",
    7: "Self-employed (own incorporated business)",
    8: "Working without pay in family business/farm",
}

# SCHL - Educational attainment
SCHL_LABELS = {
    1: "No schooling completed", 2: "Nursery / preschool", 3: "Kindergarten",
    4: "Grade 1", 5: "Grade 2", 6: "Grade 3", 7: "Grade 4", 8: "Grade 5",
    9: "Grade 6", 10: "Grade 7", 11: "Grade 8", 12: "Grade 9", 13: "Grade 10",
    14: "Grade 11", 15: "12th grade - no diploma", 16: "High school diploma",
    17: "GED / alternative credential", 18: "Some college, < 1 year",
    19: "Some college, 1+ years, no degree", 20: "Associate's degree",
    21: "Bachelor's degree", 22: "Master's degree",
    23: "Professional degree (beyond bachelor's)", 24: "Doctorate degree",
}

# MAR - Marital status
MAR_LABELS = {
    1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated",
    5: "Never married",
}

# RELSHIPP - Relationship to the householder (reference person)
RELSHIPP_LABELS = {
    20: "Reference person (householder)", 21: "Opposite-sex spouse",
    22: "Opposite-sex unmarried partner", 23: "Same-sex spouse",
    24: "Same-sex unmarried partner", 25: "Biological child", 26: "Adopted child",
    27: "Stepchild", 28: "Sibling", 29: "Parent", 30: "Grandchild",
    31: "Parent-in-law", 32: "Child-in-law", 33: "Other relative",
    34: "Roommate / housemate", 35: "Foster child", 36: "Other non-relative",
    38: "Institutionalized group-quarters resident",
}

# SEX
SEX_LABELS = {1: "Male", 2: "Female"}

# RAC1P - Race (recoded, single)
RAC1P_LABELS = {
    1: "White alone", 2: "Black / African American alone",
    3: "American Indian alone", 4: "Alaska Native alone",
    5: "American Indian / Alaska Native (other)", 6: "Asian alone",
    7: "Native Hawaiian / Pacific Islander alone", 8: "Some other race alone",
    9: "Two or more races",
}

# HISP - Hispanic origin
HISP_LABELS = {
    1: "Not Hispanic / Latino", 2: "Mexican", 3: "Puerto Rican", 4: "Cuban",
    5: "Dominican", 6: "Costa Rican", 7: "Guatemalan", 8: "Honduran",
    9: "Nicaraguan", 10: "Panamanian", 11: "Salvadoran", 12: "Other Central American",
    13: "Argentinean", 14: "Bolivian", 15: "Chilean", 16: "Colombian",
    17: "Ecuadorian", 18: "Paraguayan", 19: "Peruvian", 20: "Uruguayan",
    21: "Venezuelan", 22: "Other South American", 23: "Spaniard",
    24: "Other Spanish / Hispanic / Latino",
}

# CIT - Citizenship status
CIT_LABELS = {
    1: "Born in the United States",
    2: "Born in U.S. territory (PR, Guam, USVI, etc.)",
    3: "Born abroad to U.S. citizen parent(s)",
    4: "U.S. citizen by naturalization",
    5: "Not a U.S. citizen",
}

# NATIVITY
NATIVITY_LABELS = {1: "Native", 2: "Foreign-born"}

# ENG - Ability to speak English
ENG_LABELS = {
    0: "Speaks only English (N/A)", 1: "Very well", 2: "Well",
    3: "Not well", 4: "Not at all",
}

# DIS - Disability status
DIS_LABELS = {1: "With a disability", 2: "Without a disability"}

# ST / POBP US births - state FIPS codes
STATE_FIPS = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
    12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
    18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
    23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
    27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
    31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
    35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota",
    39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota", 47: "Tennessee",
    48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
}
ST_LABELS = dict(STATE_FIPS)

# OCCP - Occupation (high-cardinality). FULL code->title map from the Census ACS
# PUMS 2024 Data Dictionary, regenerated by demo_app/gen_labels.py. Every valid
# code the model knows now resolves to a readable occupation title.
OCCP_LABELS = {
    10: "Chief Executives And Legislators",
    20: "General And Operations Managers",
    40: "Advertising And Promotions Managers",
    51: "Marketing Managers",
    52: "Sales Managers",
    60: "Public Relations And Fundraising Managers",
    101: "Administrative Services Managers",
    102: "Facilities Managers",
    110: "Computer And Information Systems Managers",
    120: "Financial Managers",
    135: "Compensation And Benefits Managers",
    136: "Human Resources Managers",
    137: "Training And Development Managers",
    140: "Industrial Production Managers",
    150: "Purchasing Managers",
    160: "Transportation, Storage, And Distribution Managers",
    205: "Farmers, Ranchers, And Other Agricultural Managers",
    220: "Construction Managers",
    230: "Education And Childcare Administrators",
    300: "Architectural And Engineering Managers",
    310: "Food Service Managers",
    335: "Entertainment and Recreation Managers",
    340: "Lodging Managers",
    350: "Medical And Health Services Managers",
    360: "Natural Sciences Managers",
    410: "Property, Real Estate, And Community Association Managers",
    420: "Social And Community Service Managers",
    425: "Emergency Management Directors",
    440: "Other Managers",
    500: "Agents And Business Managers Of Artists, Performers, And Athletes",
    510: "Buyers And Purchasing Agents, Farm Products",
    520: "Wholesale And Retail Buyers, Except Farm Products",
    530: "Purchasing Agents, Except Wholesale, Retail, And Farm Products",
    540: "Claims Adjusters, Appraisers, Examiners, And Investigators",
    565: "Compliance Officers",
    600: "Cost Estimators",
    630: "Human Resources Workers",
    640: "Compensation, Benefits, And Job Analysis Specialists",
    650: "Training And Development Specialists",
    700: "Logisticians",
    705: "Project Management Specialists",
    710: "Management Analysts",
    725: "Meeting, Convention, And Event Planners",
    726: "Fundraisers",
    735: "Market Research Analysts And Marketing Specialists",
    750: "Business Operations Specialists, All Other",
    800: "Accountants And Auditors",
    810: "Property Appraisers and Assessors",
    820: "Budget Analysts",
    830: "Credit Analysts",
    845: "Financial And Investment Analysts",
    850: "Personal Financial Advisors",
    860: "Insurance Underwriters",
    900: "Financial Examiners",
    910: "Credit Counselors And Loan Officers",
    930: "Tax Examiners And Collectors, And Revenue Agents",
    940: "Tax Preparers",
    960: "Other Financial Specialists",
    1005: "Computer And Information Research Scientists",
    1006: "Computer Systems Analysts",
    1007: "Information Security Analysts",
    1010: "Computer Programmers",
    1021: "Software Developers",
    1022: "Software Quality Assurance Analysts and Testers",
    1031: "Web Developers",
    1032: "Web And Digital Interface Designers",
    1050: "Computer Support Specialists",
    1065: "Database Administrators and Architects",
    1105: "Network And Computer Systems Administrators",
    1106: "Computer Network Architects",
    1108: "Computer Occupations, All Other",
    1200: "Actuaries",
    1220: "Operations Research Analysts",
    1240: "Other Mathematical Science Occupations",
    1305: "Architects, Except Landscape And Naval",
    1306: "Landscape Architects",
    1310: "Surveyors, Cartographers, And Photogrammetrists",
    1320: "Aerospace Engineers",
    1340: "Biomedical And Agricultural Engineers",
    1350: "Chemical Engineers",
    1360: "Civil Engineers",
    1400: "Computer Hardware Engineers",
    1410: "Electrical And Electronics Engineers",
    1420: "Environmental Engineers",
    1430: "Industrial Engineers, Including Health And Safety",
    1440: "Marine Engineers And Naval Architects",
    1450: "Materials Engineers",
    1460: "Mechanical Engineers",
    1520: "Petroleum, Mining And Geological Engineers, Including Mining Safety Engineers",
    1530: "Other Engineers",
    1541: "Architectural And Civil Drafters",
    1545: "Other Drafters",
    1551: "Electrical And Electronic Engineering Technologists And Technicians",
    1555: "Other Engineering Technologists And Technicians, Except Drafters",
    1560: "Surveying And Mapping Technicians",
    1600: "Agricultural And Food Scientists",
    1610: "Biological Scientists",
    1640: "Conservation Scientists And Foresters",
    1650: "Other Life Scientists",
    1700: "Astronomers And Physicists",
    1710: "Atmospheric And Space Scientists",
    1720: "Chemists And Materials Scientists",
    1745: "Environmental Scientists And Specialists, Including Health",
    1750: "Geoscientists And Hydrologists, Except Geographers",
    1760: "Physical Scientists, All Other",
    1800: "Economists",
    1821: "Clinical And Counseling Psychologists",
    1822: "School Psychologists",
    1825: "Other Psychologists",
    1840: "Urban And Regional Planners",
    1860: "Other Social Scientists",
    1900: "Agricultural And Food Science Technicians",
    1910: "Biological Technicians",
    1920: "Chemical Technicians",
    1935: "Environmental Science and Geoscience Technicians, And Nuclear Technicians",
    1970: "Other Life, Physical, And Social Science Technicians",
    1980: "Occupational Health And Safety Specialists and Technicians",
    2001: "Substance Abuse And Behavioral Disorder Counselors",
    2002: "Educational, Guidance, And Career Counselors And Advisors",
    2003: "Marriage And Family Therapists",
    2004: "Mental Health Counselors",
    2005: "Rehabilitation Counselors",
    2006: "Counselors, All Other",
    2011: "Child, Family, And School Social Workers",
    2012: "Healthcare Social Workers",
    2013: "Mental Health And Substance Abuse Social Workers",
    2014: "Social Workers, All Other",
    2015: "Probation Officers And Correctional Treatment Specialists",
    2016: "Social And Human Service Assistants",
    2025: "Other Community and Social Service Specialists",
    2040: "Clergy",
    2050: "Directors, Religious Activities And Education",
    2060: "Religious Workers, All Other",
    2100: "Lawyers, And Judges, Magistrates, And Other Judicial Workers",
    2105: "Judicial Law Clerks",
    2145: "Paralegals And Legal Assistants",
    2170: "Title Examiners, Abstractors, and Searchers",
    2180: "Legal Support Workers, All Other",
    2205: "Postsecondary Teachers",
    2300: "Preschool And Kindergarten Teachers",
    2310: "Elementary And Middle School Teachers",
    2320: "Secondary School Teachers",
    2330: "Special Education Teachers",
    2350: "Tutors",
    2360: "Other Teachers and Instructors",
    2400: "Archivists, Curators, And Museum Technicians",
    2435: "Librarians And Media Collections Specialists",
    2440: "Library Technicians",
    2545: "Teaching Assistants",
    2555: "Other Educational Instruction and Library Workers",
    2600: "Artists And Related Workers",
    2631: "Commercial And Industrial Designers",
    2632: "Fashion Designers",
    2633: "Floral Designers",
    2634: "Graphic Designers",
    2635: "Interior Designers",
    2636: "Merchandise Displayers And Window Trimmers",
    2640: "Other Designers",
    2700: "Actors",
    2710: "Producers And Directors",
    2721: "Athletes and Sports Competitors",
    2722: "Coaches and Scouts",
    2723: "Umpires, Referees, And Other Sports Officials",
    2740: "Dancers And Choreographers",
    2751: "Music Directors and Composers",
    2752: "Musicians and Singers",
    2755: "Disc Jockeys, Except Radio",
    2770: "Entertainers And Performers, Sports and Related Workers, All Other",
    2805: "Broadcast Announcers And Radio Disc Jockeys",
    2810: "News Analysts, Reporters, And Journalists",
    2825: "Public Relations Specialists",
    2830: "Editors",
    2840: "Technical Writers",
    2850: "Writers And Authors",
    2861: "Interpreters and Translators",
    2862: "Court Reporters and Simultaneous Captioners",
    2865: "Media And Communication Workers, All Other",
    2905: "Other Media And Communication Equipment Workers",
    2910: "Photographers",
    2920: "Television, Video, And Film Camera Operators And Editors",
    3000: "Chiropractors",
    3010: "Dentists",
    3030: "Dietitians And Nutritionists",
    3040: "Optometrists",
    3050: "Pharmacists",
    3090: "Physicians",
    3100: "Surgeons",
    3110: "Physician Assistants",
    3120: "Podiatrists",
    3140: "Audiologists",
    3150: "Occupational Therapists",
    3160: "Physical Therapists",
    3200: "Radiation Therapists",
    3210: "Recreational Therapists",
    3220: "Respiratory Therapists",
    3230: "Speech-Language Pathologists",
    3245: "Other Therapists",
    3250: "Veterinarians",
    3255: "Registered Nurses",
    3256: "Nurse Anesthetists",
    3258: "Nurse Practitioners, And Nurse Midwives",
    3261: "Acupuncturists",
    3270: "Healthcare Diagnosing Or Treating Practitioners, All Other",
    3300: "Clinical Laboratory Technologists And Technicians",
    3310: "Dental Hygienists",
    3321: "Cardiovascular Technologists and Technicians",
    3322: "Diagnostic Medical Sonographers",
    3323: "Radiologic Technologists And Technicians",
    3324: "Magnetic Resonance Imaging Technologists",
    3330: "Nuclear Medicine Technologists and Medical Dosimetrists",
    3401: "Emergency Medical Technicians",
    3402: "Paramedics",
    3421: "Pharmacy Technicians",
    3422: "Psychiatric Technicians",
    3423: "Surgical Technologists",
    3424: "Veterinary Technologists and Technicians",
    3430: "Dietetic Technicians And Ophthalmic Medical Technicians",
    3500: "Licensed Practical And Licensed Vocational Nurses",
    3515: "Medical Records Specialists",
    3520: "Opticians, Dispensing",
    3545: "Miscellaneous Health Technologists and Technicians",
    3550: "Other Healthcare Practitioners and Technical Occupations",
    3601: "Home Health Aides",
    3602: "Personal Care Aides",
    3603: "Nursing Assistants",
    3605: "Orderlies and Psychiatric Aides",
    3610: "Occupational Therapy Assistants And Aides",
    3620: "Physical Therapist Assistants And Aides",
    3630: "Massage Therapists",
    3640: "Dental Assistants",
    3645: "Medical Assistants",
    3646: "Medical Transcriptionists",
    3647: "Pharmacy Aides",
    3648: "Veterinary Assistants And Laboratory Animal Caretakers",
    3649: "Phlebotomists",
    3655: "Other Healthcare Support Workers",
    3700: "First-Line Supervisors Of Correctional Officers",
    3710: "First-Line Supervisors Of Police And Detectives",
    3720: "First-Line Supervisors Of Firefighting And Prevention Workers",
    3725: "Miscellaneous First-Line Supervisors, Protective Service Workers",
    3740: "Firefighters",
    3750: "Fire Inspectors",
    3801: "Bailiffs",
    3802: "Correctional Officers and Jailers",
    3820: "Detectives And Criminal Investigators",
    3840: "Fish And Game Wardens And Parking Enforcement Officers",
    3870: "Police Officers",
    3900: "Animal Control Workers",
    3910: "Private Detectives And Investigators",
    3930: "Security Guards And Gambling Surveillance Officers",
    3940: "Crossing Guards And Flaggers",
    3945: "Transportation Security Screeners",
    3946: "School Bus Monitors",
    3960: "Other Protective Service Workers",
    4000: "Chefs And Head Cooks",
    4010: "First-Line Supervisors Of Food Preparation And Serving Workers",
    4020: "Cooks",
    4030: "Food Preparation Workers",
    4040: "Bartenders",
    4055: "Fast Food And Counter Workers",
    4110: "Waiters And Waitresses",
    4120: "Food Servers, Nonrestaurant",
    4130: "Dining Room And Cafeteria Attendants And Bartender Helpers",
    4140: "Dishwashers",
    4150: "Hosts And Hostesses, Restaurant, Lounge, And Coffee Shop",
    4160: "Food Preparation and Serving Related Workers, All Other",
    4200: "First-Line Supervisors Of Housekeeping And Janitorial Workers",
    4210: "First-Line Supervisors Of Landscaping, Lawn Service, And Groundskeeping Workers",
    4220: "Janitors And Building Cleaners",
    4230: "Maids And Housekeeping Cleaners",
    4240: "Pest Control Workers",
    4251: "Landscaping And Groundskeeping Workers",
    4252: "Tree Trimmers and Pruners",
    4255: "Other Grounds Maintenance Workers",
    4330: "Supervisors Of Personal Care And Service Workers",
    4340: "Animal Trainers",
    4350: "Animal Caretakers",
    4400: "Gambling Services Workers",
    4420: "Ushers, Lobby Attendants, And Ticket Takers",
    4435: "Other Entertainment Attendants And Related Workers",
    4461: "Embalmers, Crematory Operators And Funeral Attendants",
    4465: "Morticians, Undertakers, And Funeral Arrangers",
    4500: "Barbers",
    4510: "Hairdressers, Hairstylists, And Cosmetologists",
    4521: "Manicurists And Pedicurists",
    4522: "Skincare Specialists",
    4525: "Other Personal Appearance Workers",
    4530: "Baggage Porters, Bellhops, And Concierges",
    4540: "Tour And Travel Guides",
    4600: "Childcare Workers",
    4621: "Exercise Trainers And Group Fitness Instructors",
    4622: "Recreation Workers",
    4640: "Residential Advisors",
    4655: "Personal Care and Service Workers, All Other",
    4700: "First-Line Supervisors Of Retail Sales Workers",
    4710: "First-Line Supervisors Of Non-Retail Sales Workers",
    4720: "Cashiers",
    4740: "Counter And Rental Clerks",
    4750: "Parts Salespersons",
    4760: "Retail Salespersons",
    4800: "Advertising Sales Agents",
    4810: "Insurance Sales Agents",
    4820: "Securities, Commodities, And Financial Services Sales Agents",
    4830: "Travel Agents",
    4840: "Sales Representatives Of Services, Except Advertising, Insurance, Financial Services, And Travel",
    4850: "Sales Representatives, Wholesale And Manufacturing",
    4900: "Models, Demonstrators, And Product Promoters",
    4920: "Real Estate Brokers And Sales Agents",
    4930: "Sales Engineers",
    4940: "Telemarketers",
    4950: "Door-To-Door Sales Workers, News And Street Vendors, And Related Workers",
    4965: "Sales And Related Workers, All Other",
    5000: "First-Line Supervisors Of Office And Administrative Support Workers",
    5010: "Switchboard Operators, Including Answering Service",
    5020: "Telephone Operators",
    5040: "Communications Equipment Operators, All Other",
    5100: "Bill And Account Collectors",
    5110: "Billing And Posting Clerks",
    5120: "Bookkeeping, Accounting, And Auditing Clerks",
    5140: "Payroll And Timekeeping Clerks",
    5150: "Procurement Clerks",
    5160: "Tellers",
    5165: "Other Financial Clerks",
    5220: "Court, Municipal, And License Clerks",
    5230: "Credit Authorizers, Checkers, And Clerks",
    5240: "Customer Service Representatives",
    5250: "Eligibility Interviewers, Government Programs",
    5260: "File Clerks",
    5300: "Hotel, Motel, And Resort Desk Clerks",
    5310: "Interviewers, Except Eligibility And Loan",
    5320: "Library Assistants, Clerical",
    5330: "Loan Interviewers And Clerks",
    5340: "New Accounts Clerks",
    5350: "Correspondence Clerks And Order Clerks",
    5360: "Human Resources Assistants, Except Payroll And Timekeeping",
    5400: "Receptionists And Information Clerks",
    5410: "Reservation And Transportation Ticket Agents And Travel Clerks",
    5420: "Other Information And Records Clerks",
    5500: "Cargo And Freight Agents",
    5510: "Couriers And Messengers",
    5521: "Public Safety Telecommunicators",
    5522: "Dispatchers, Except Police, Fire, And Ambulance",
    5530: "Meter Readers, Utilities",
    5540: "Postal Service Clerks",
    5550: "Postal Service Mail Carriers",
    5560: "Postal Service Mail Sorters, Processors, And Processing Machine Operators",
    5600: "Production, Planning, And Expediting Clerks",
    5610: "Shipping, Receiving, And Inventory Clerks",
    5630: "Weighers, Measurers, Checkers, And Samplers, Recordkeeping",
    5710: "Executive Secretaries And Executive Administrative Assistants",
    5720: "Legal Secretaries and Administrative Assistants",
    5730: "Medical Secretaries and Administrative Assistants",
    5740: "Secretaries And Administrative Assistants, Except Legal, Medical, And Executive",
    5810: "Data Entry Keyers",
    5820: "Word Processors And Typists",
    5840: "Insurance Claims And Policy Processing Clerks",
    5850: "Mail Clerks And Mail Machine Operators, Except Postal Service",
    5860: "Office Clerks, General",
    5900: "Office Machine Operators, Except Computer",
    5910: "Proofreaders And Copy Markers",
    5920: "Statistical Assistants",
    5940: "Other Office And Administrative Support Workers",
    6005: "First-Line Supervisors Of Farming, Fishing, And Forestry Workers",
    6010: "Agricultural Inspectors",
    6040: "Graders And Sorters, Agricultural Products",
    6050: "Other Agricultural Workers",
    6115: "Fishing And Hunting Workers",
    6120: "Forest And Conservation Workers",
    6130: "Logging Workers",
    6200: "First-Line Supervisors Of Construction Trades And Extraction Workers",
    6210: "Boilermakers",
    6220: "Brickmasons, Blockmasons, Stonemasons, And Reinforcing Iron And Rebar Workers",
    6230: "Carpenters",
    6240: "Carpet, Floor, And Tile Installers And Finishers",
    6250: "Cement Masons, Concrete Finishers, And Terrazzo Workers",
    6260: "Construction Laborers",
    6305: "Construction Equipment Operators",
    6330: "Drywall Installers, Ceiling Tile Installers, And Tapers",
    6355: "Electricians",
    6360: "Glaziers",
    6400: "Insulation Workers",
    6410: "Painters and Paperhangers",
    6441: "Pipelayers",
    6442: "Plumbers, Pipefitters, And Steamfitters",
    6460: "Plasterers And Stucco Masons",
    6515: "Roofers",
    6520: "Sheet Metal Workers",
    6530: "Structural Iron And Steel Workers",
    6540: "Solar Photovoltaic Installers",
    6600: "Helpers, Construction Trades",
    6660: "Construction And Building Inspectors",
    6700: "Elevator And Escalator Installers And Repairers",
    6710: "Fence Erectors",
    6720: "Hazardous Materials Removal Workers",
    6730: "Highway Maintenance Workers",
    6740: "Rail-Track Laying And Maintenance Equipment Operators",
    6765: "Other Construction And Related Workers",
    6800: "Derrick, Rotary Drill, And Service Unit Operators, And Roustabouts, Oil And Gas",
    6825: "Surface Mining Machine Operators And Earth Drillers",
    6835: "Explosives Workers, Ordnance Handling Experts, and Blasters",
    6850: "Underground Mining Machine Operators",
    6950: "Other Extraction Workers",
    7000: "First-Line Supervisors Of Mechanics, Installers, And Repairers",
    7010: "Computer, Automated Teller, And Office Machine Repairers",
    7020: "Radio And Telecommunications Equipment Installers And Repairers",
    7030: "Avionics Technicians",
    7040: "Electric Motor, Power Tool, And Related Repairers",
    7100: "Other Electrical And Electronic Equipment Mechanics, Installers, And Repairers",
    7120: "Audiovisual Equipment Installers And Repairers",
    7130: "Security And Fire Alarm Systems Installers",
    7140: "Aircraft Mechanics And Service Technicians",
    7150: "Automotive Body And Related Repairers",
    7160: "Automotive Glass Installers And Repairers",
    7200: "Automotive Service Technicians And Mechanics",
    7210: "Bus And Truck Mechanics And Diesel Engine Specialists",
    7220: "Heavy Vehicle And Mobile Equipment Service Technicians And Mechanics",
    7240: "Small Engine Mechanics",
    7260: "Miscellaneous Vehicle And Mobile Equipment Mechanics, Installers, And Repairers",
    7300: "Control And Valve Installers And Repairers",
    7315: "Heating, Air Conditioning, And Refrigeration Mechanics And Installers",
    7320: "Home Appliance Repairers",
    7330: "Industrial And Refractory Machinery Mechanics",
    7340: "Maintenance And Repair Workers, General",
    7350: "Maintenance Workers, Machinery",
    7360: "Millwrights",
    7410: "Electrical Power-Line Installers And Repairers",
    7420: "Telecommunications Line Installers And Repairers",
    7430: "Precision Instrument And Equipment Repairers",
    7510: "Coin, Vending, And Amusement Machine Servicers And Repairers",
    7540: "Locksmiths And Safe Repairers",
    7560: "Riggers",
    7610: "Helpers--Installation, Maintenance, And Repair Workers",
    7640: "Other Installation, Maintenance, And Repair Workers",
    7700: "First-Line Supervisors Of Production And Operating Workers",
    7720: "Electrical, Electronics, And Electromechanical Assemblers",
    7730: "Engine And Other Machine Assemblers",
    7740: "Structural Metal Fabricators And Fitters",
    7750: "Other Assemblers And Fabricators",
    7800: "Bakers",
    7810: "Butchers And Other Meat, Poultry, And Fish Processing Workers",
    7830: "Food And Tobacco Roasting, Baking, And Drying Machine Operators And Tenders",
    7840: "Food Batchmakers",
    7850: "Food Cooking Machine Operators And Tenders",
    7855: "Food Processing Workers, All Other",
    7905: "Computer Numerically Controlled Tool Operators And Programmers",
    7925: "Forming Machine Setters, Operators, And Tenders, Metal And Plastic",
    7950: "Cutting, Punching, And Press Machine Setters, Operators, And Tenders, Metal And Plastic",
    8000: "Grinding, Lapping, Polishing, And Buffing Machine Tool Setters, Operators, And Tenders, Metal And Plastic",
    8025: "Other Machine Tool Setters, Operators, And Tenders, Metal and Plastic",
    8030: "Machinists",
    8040: "Metal Furnace Operators, Tenders, Pourers, And Casters",
    8100: "Model Makers, Patternmakers, And Molding Machine Setters, Metal And Plastic",
    8130: "Tool And Die Makers",
    8140: "Welding, Soldering, And Brazing Workers",
    8225: "Other Metal Workers And Plastic Workers",
    8250: "Prepress Technicians And Workers",
    8255: "Printing Press Operators",
    8256: "Print Binding And Finishing Workers",
    8300: "Laundry And Dry-Cleaning Workers",
    8310: "Pressers, Textile, Garment, And Related Materials",
    8320: "Sewing Machine Operators",
    8335: "Shoe And Leather Workers",
    8350: "Tailors, Dressmakers, And Sewers",
    8365: "Textile Machine Setters, Operators, And Tenders",
    8450: "Upholsterers",
    8465: "Other Textile, Apparel, And Furnishings Workers",
    8500: "Cabinetmakers And Bench Carpenters",
    8510: "Furniture Finishers",
    8530: "Sawing Machine Setters, Operators, And Tenders, Wood",
    8540: "Woodworking Machine Setters, Operators, And Tenders, Except Sawing",
    8555: "Other Woodworkers",
    8600: "Power Plant Operators, Distributors, And Dispatchers",
    8610: "Stationary Engineers And Boiler Operators",
    8620: "Water And Wastewater Treatment Plant And System Operators",
    8630: "Miscellaneous Plant And System Operators",
    8640: "Chemical Processing Machine Setters, Operators, And Tenders",
    8650: "Crushing, Grinding, Polishing, Mixing, And Blending Workers",
    8710: "Cutting Workers",
    8720: "Extruding, Forming, Pressing, And Compacting Machine Setters, Operators, And Tenders",
    8730: "Furnace, Kiln, Oven, Drier, And Kettle Operators And Tenders",
    8740: "Inspectors, Testers, Sorters, Samplers, And Weighers",
    8750: "Jewelers And Precious Stone And Metal Workers",
    8760: "Dental And Ophthalmic Laboratory Technicians And Medical Appliance Technicians",
    8800: "Packaging And Filling Machine Operators And Tenders",
    8810: "Painting Workers",
    8830: "Photographic Process Workers And Processing Machine Operators",
    8850: "Adhesive Bonding Machine Operators And Tenders",
    8910: "Etchers And Engravers",
    8920: "Molders, Shapers, And Casters, Except Metal And Plastic",
    8930: "Paper Goods Machine Setters, Operators, And Tenders",
    8940: "Tire Builders",
    8950: "Helpers--Production Workers",
    8990: "Miscellaneous Production Workers, Including Equipment Operators And Tenders",
    9005: "Supervisors Of Transportation And Material Moving Workers",
    9030: "Aircraft Pilots And Flight Engineers",
    9040: "Air Traffic Controllers And Airfield Operations Specialists",
    9050: "Flight Attendants",
    9110: "Ambulance Drivers And Attendants, Except Emergency Medical Technicians",
    9121: "Bus Drivers, School",
    9122: "Bus Drivers, Transit And Intercity",
    9130: "Driver/Sales Workers And Truck Drivers",
    9141: "Shuttle Drivers And Chauffeurs",
    9142: "Taxi Drivers",
    9150: "Motor Vehicle Operators, All Other",
    9210: "Locomotive Engineers And Operators",
    9240: "Railroad Conductors And Yardmasters",
    9265: "Other Rail Transportation Workers",
    9300: "Sailors And Marine Oilers, And Ship Engineers",
    9310: "Ship And Boat Captains And Operators",
    9350: "Parking Attendants",
    9365: "Transportation Service Attendants",
    9410: "Transportation Inspectors",
    9415: "Passenger Attendants",
    9430: "Other Transportation Workers",
    9510: "Crane And Tower Operators",
    9570: "Conveyor, Dredge, And Hoist and Winch Operators",
    9600: "Industrial Truck And Tractor Operators",
    9610: "Cleaners Of Vehicles And Equipment",
    9620: "Laborers And Freight, Stock, And Material Movers, Hand",
    9630: "Machine Feeders And Offbearers",
    9640: "Packers And Packagers, Hand",
    9645: "Stockers And Order Fillers",
    9650: "Pumping Station Operators",
    9720: "Refuse And Recyclable Material Collectors",
    9760: "Other Material Moving Workers",
    9800: "Military Officer Special And Tactical Operations Leaders",
    9810: "First-Line Enlisted Military Supervisors",
    9825: "Military Enlisted Tactical Operations And Air/Weapons Specialists And Crew Members",
    9830: "Military, Rank Not Specified",
    9920: "Unemployed, With No Work Experience In The Last 5 Years Or Earlier Or Never Worked",
}

# INDP - Industry (high-cardinality). FULL code->title map from the Census ACS
# PUMS 2024 Data Dictionary, regenerated by demo_app/gen_labels.py.
INDP_LABELS = {
    170: "Crop Production",
    180: "Animal Production And Aquaculture",
    190: "Forestry Except Logging",
    270: "Logging",
    280: "Fishing, Hunting And Trapping",
    290: "Support Activities For Agriculture And Forestry",
    370: "Oil And Gas Extraction",
    380: "Coal Mining",
    390: "Metal Ore Mining",
    470: "Nonmetallic Mineral Mining And Quarrying",
    490: "Support Activities For Mining",
    570: "Electric Power Generation, Transmission And Distribution",
    580: "Natural Gas Distribution",
    590: "Electric And Gas, And Other Combinations",
    670: "Water Supply And Irrigation Systems, And Steam And Air- Conditioning Supply",
    680: "Sewage Treatment Facilities",
    690: "Not Specified Utilities",
    770: "Construction (The Cleaning Of Buildings And Dwellings Is Incidental During Construction And Immediately After Construction)",
    1070: "Animal Food, Grain And Oilseed Milling",
    1080: "Sugar And Confectionery Products",
    1090: "Fruit And Vegetable Preserving And Specialty Food",
    1170: "Dairy Product",
    1180: "Animal Slaughtering And Processing",
    1190: "Retail Bakeries",
    1270: "Bakeries And Tortilla, Except Retail Bakeries",
    1280: "Seafood And Other Miscellaneous Foods, N.E.C.",
    1290: "Not Specified Food Industries",
    1370: "Beverage",
    1390: "Tobacco",
    1470: "Fiber, Yarn, And Thread Mills",
    1480: "Fabric Mills, Except Knit Fabric Mills",
    1490: "Textile And Fabric Finishing And Fabric Coating Mills",
    1570: "Carpet And Rug Mills",
    1590: "Textile Product Mills, Except Carpet And Rug",
    1670: "Knit Fabric Mills, And Apparel Knitting Mills",
    1691: "Cut And Sew, And Apparel Accessories And Other Apparel",
    1770: "Footwear",
    1790: "Leather And Hide Tanning And Finishing, And Other Leather And Allied Product",
    1870: "Pulp, Paper, And Paperboard Mills",
    1880: "Paperboard Container",
    1890: "Miscellaneous Paper And Pulp Products",
    1990: "Printing And Related Support Activities",
    2070: "Petroleum Refineries",
    2090: "Petroleum And Coal Products Manufacturing, Except Petroleum Refineries",
    2170: "Resin, Synthetic Rubber, And Artificial And Synthetic Fibers And Filaments",
    2180: "Pesticide, Fertilizer, And Other Agricultural Chemical",
    2190: "Pharmaceutical And Medicine",
    2270: "Paint, Coating, And Adhesive",
    2280: "Soap, Cleaning Compound, And Toilet Preparation",
    2290: "Basic Chemical, And Other Chemical Product And Preparation Chemicals",
    2370: "Plastics Product",
    2380: "Tire",
    2390: "Rubber Products, Except Tires",
    2470: "Pottery, Ceramics, And Plumbing Fixture",
    2480: "Clay Building Material And Refractories",
    2490: "Glass And Glass Product",
    2570: "Cement, Concrete, Lime, And Gypsum Product",
    2590: "Other Nonmetallic Mineral Product",
    2670: "Iron And Steel Mills And Steel Product",
    2680: "Alumina And Aluminum Production And Processing",
    2690: "Nonferrous Metal (Except Aluminum) Production And Processing",
    2770: "Foundries",
    2780: "Forgings And Stamping",
    2790: "Cutlery And Handtool",
    2870: "Architectural And Structural Metals, And Boiler, Tank, And Shipping Container",
    2880: "Machine Shops; Turned Product; Screw, Nut, And Bolt",
    2890: "Coating, Engraving, Heat Treating, And Allied Activities",
    2970: "Ordnance",
    2980: "Miscellaneous Fabricated Metal Product",
    2990: "Not Specified Metal Industries",
    3070: "Agricultural Implement",
    3080: "Construction, And Mining And Oil And Gas Field Machinery",
    3095: "Commercial And Service Industry Machinery",
    3170: "Metalworking Machinery",
    3180: "Engine, Turbine, And Power Transmission Equipment",
    3291: "Machinery, N.E.C. Or Not Specified",
    3365: "Computer And Peripheral Equipment",
    3370: "Communications, Audio, And Video Equipment",
    3380: "Navigational, Measuring, Electromedical, And Control Instruments",
    3390: "Semiconductor, Magnetic And Optical Media, And Other Electronic Component",
    3470: "Household Appliance",
    3490: "Electric Lighting And Electrical Equipment, And Other Electrical Component, N.E.C.",
    3570: "Motor Vehicles And Motor Vehicle Equipment",
    3580: "Aircraft, Aircraft Engine, And Aircraft Parts",
    3590: "Guided Missile And Space Vehicle And Parts",
    3670: "Railroad Rolling Stock",
    3680: "Ship And Boat Building",
    3690: "Other Transportation Equipment",
    3770: "Sawmills And Wood Preservation",
    3780: "Veneer, Plywood, And Engineered Wood Product",
    3790: "Manufactured Home (Mobile Home) And Prefabricated Wood Building",
    3875: "Miscellaneous Wood Product",
    3895: "Furniture And Related Product",
    3960: "Medical Equipment And Supplies",
    3970: "Sporting And Athletic Goods, And Doll, Toy And Game",
    3980: "Miscellaneous Manufacturing, N.E.C.",
    3990: "Not Specified Manufacturing Industries",
    4070: "Motor Vehicle And Motor Vehicle Parts And Supplies Merchant Wholesalers",
    4080: "Furniture And Home Furnishing Merchant Wholesalers",
    4090: "Lumber And Other Construction Materials Merchant Wholesalers",
    4170: "Professional And Commercial Equipment And Supplies Merchant Wholesalers",
    4180: "Metal And Mineral, Except Petroleum, Merchant Wholesalers",
    4195: "Household Appliances And Electrical And Electronic Goods Merchant Wholesalers",
    4265: "Hardware, And Plumbing And Heating Equipment, And Supplies Merchant Wholesalers",
    4270: "Machinery, Equipment, And Supplies Merchant Wholesalers",
    4280: "Recyclable Material Merchant Wholesalers",
    4290: "Miscellaneous Durable Goods, Except Recyclable Material, Merchant Wholesalers",
    4370: "Paper And Paper Product Merchant Wholesalers",
    4380: "Drugs, Druggists' Sundries, And Chemical And Allied Products Merchant Wholesalers",
    4390: "Apparel, Piece Goods, And Notions Merchant Wholesalers",
    4470: "Grocery And Related Product Merchant Wholesalers",
    4480: "Farm Product Raw Material Merchant Wholesalers",
    4490: "Petroleum And Petroleum Products Merchant Wholesalers",
    4560: "Beer, Wine, And Distilled Alcoholic Beverage Merchant Wholesalers",
    4570: "Farm Supplies Merchant Wholesalers",
    4580: "Miscellaneous Nondurable Goods Merchant, Except Farm Supplies, Wholesalers",
    4585: "Wholesale Trade Agents And Brokers",
    4590: "Not Specified Wholesale Trade",
    4670: "Automobile Dealers",
    4681: "Other Motor Vehicle Dealers",
    4691: "Automotive Parts, Accessories, And Tire Retailers",
    4771: "Furniture And Home Furnishings Retailers",
    4796: "Electronics And Appliance Retailers",
    4871: "Building Material And Supplies Dealers, Except Hardware Retailers",
    4881: "Hardware Retailers",
    4891: "Lawn And Garden Equipment And Supplies Retailers",
    4971: "Supermarkets And Other Grocery (Except Convenience) Retailers",
    4973: "Convenience Retailers And Vending Machine Operators",
    4981: "Specialty Food Retailers",
    4991: "Beer, Wine, And Liquor Retailers",
    5071: "Pharmacies And Drug Retailers",
    5081: "Health And Personal Care, Except Pharmacies And Drug, Retailers",
    5090: "Gasoline Stations",
    5171: "Clothing And Clothing Accessories Retailers",
    5181: "Shoe Retailers",
    5191: "Jewelry, Luggage, And Leather Goods Retailers",
    5276: "Sporting Goods, And Hobby, Toy, And Game Retailers",
    5281: "Sewing, Needlework, And Piece Goods Retailers",
    5296: "Musical Instrument And Supplies Retailers",
    5371: "Book Retailers And News Dealers",
    5382: "Department Stores",
    5392: "Warehouse Clubs, Supercenters, And Other General Merchandise Retailers",
    5471: "Florists",
    5481: "Office Supplies And Stationery Retailers",
    5491: "Used Merchandise Retailers",
    5571: "Gift, Novelty, And Souvenir Retailers",
    5581: "Other Miscellaneous Retailers",
    5680: "Fuel Dealers",
    5791: "Not Specified Retail Trade",
    6070: "Air Transportation",
    6080: "Rail Transportation",
    6090: "Water Transportation",
    6170: "Truck Transportation",
    6180: "Transit And Ground Passenger Transportation, Except Taxi And Limousine Service",
    6190: "Taxi And Limousine Service",
    6270: "Pipeline Transportation",
    6280: "Scenic And Sightseeing Transportation",
    6290: "Support Activities For Transportation",
    6370: "Postal Service",
    6380: "Couriers And Messengers",
    6390: "Warehousing And Storage",
    6471: "Newspaper Publishers",
    6481: "Periodical, Book, And Directory And Mailing List, And Other Publishers",
    6490: "Software Publishers",
    6570: "Motion Pictures And Video Industries",
    6590: "Sound Recording Industries",
    6671: "Broadcasting And Content Providers",
    6680: "Wired Telecommunications Carriers",
    6690: "Telecommunications, Except Wired Telecommunications Carriers",
    6695: "Computing Infrastructure Providers, Data Processing, Web Hosting, And Related Services",
    6770: "Libraries And Archives",
    6781: "Web Search Portals And All Other Information Services",
    6871: "Monetary Authorities-Central Bank, And Commercial Banking",
    6881: "Credit Unions, And Savings Institutions And Other Depository Credit Intermediation",
    6890: "Nondepository Credit Intermediation And Related Activities",
    6970: "Securities, Commodities, Funds, Trusts, And Other Financial Investments",
    6991: "Insurance Carriers",
    6992: "Agencies, Brokerages, And Other Insurance Related Activities",
    7071: "Lessors Of Real Estate, And Offices Of Real Estate Agents And Brokers",
    7072: "Activities Related To Real Estate",
    7080: "Automotive Equipment Rental And Leasing",
    7181: "Consumer Goods Rental, And General Rental Centers",
    7190: "Commercial And Industrial Machinery and Equipment, And Nonfinancial Assets Rental And Leasing",
    7270: "Legal Services",
    7280: "Accounting, Tax Preparation, Bookkeeping, And Payroll Services",
    7290: "Architectural, Engineering, And Related Services",
    7370: "Specialized Design Services",
    7380: "Computer Systems Design And Related Services",
    7390: "Management, Scientific, And Technical Consulting Services",
    7460: "Scientific Research And Development Services",
    7470: "Advertising, Public Relations, And Related Services",
    7480: "Veterinary Services",
    7490: "Other Professional, Scientific, And Technical, Except Veterinary, Services",
    7570: "Management Of Companies And Enterprises",
    7580: "Employment Services",
    7590: "Business Support Services",
    7670: "Travel Arrangements And Reservation Services",
    7680: "Investigation And Security Services",
    7690: "Services To Buildings And Dwellings, Except Landscaping Services",
    7770: "Landscaping Services",
    7780: "Other Administrative And Other Support Services",
    7790: "Waste Management And Remediation Services",
    7860: "Elementary And Secondary Schools",
    7870: "Junior Colleges, Colleges, Universities, And Professional Schools",
    7880: "Business, Trade And Technical Schools, And Computer And Management Training",
    7890: "Other Schools And Instruction, And Educational Support Services",
    7970: "Offices Of Physicians",
    7980: "Offices Of Dentists",
    7990: "Offices Of Chiropractors",
    8070: "Offices Of Optometrists",
    8080: "Offices Of Other Health Practitioners",
    8090: "Outpatient Care Centers",
    8170: "Home Health Care Services",
    8180: "Other Health Care Services",
    8191: "General Medical And Surgical Hospitals, And Specialty (Except Psychiatric And Substance Abuse) Hospitals",
    8192: "Psychiatric And Substance Abuse Hospitals",
    8270: "Nursing Care Facilities (Skilled Nursing Facilities)",
    8290: "Residential Care Facilities, Except Skilled Nursing Facilities",
    8370: "Individual And Family Services",
    8380: "Community Food And Housing, And Emergency And Other Relief Services",
    8390: "Vocational Rehabilitation Services",
    8470: "Child Care Services",
    8561: "Performing Arts Companies",
    8562: "Spectator Sports",
    8563: "Promoters Of Performing Arts, Sports, And Similar Events, Agents And Managers For Artists, Athletes, Entertainers, And Other Public Figures",
    8564: "Independent Artists, Writers, And Performers",
    8570: "Museums, Historical Sites, And Similar Institutions",
    8580: "Bowling Centers",
    8590: "Other Amusement, Gambling, And Recreation Industries",
    8660: "Traveler Accommodation",
    8670: "Recreational Vehicle Parks And Camps, And Rooming And Boarding Houses, Dormitories, And Workers' Camps",
    8680: "Food Services And Drinking Places, except Alcoholic Beverages",
    8690: "Drinking Places (Alcoholic Beverages)",
    8770: "Automotive Repair And Maintenance, Except Car Washes",
    8780: "Car Washes",
    8790: "Electronic And Precision Equipment Repair And Maintenance",
    8870: "Commercial And Industrial Machinery And Equipment Repair And Maintenance",
    8891: "Personal And Household Goods Repair And Maintenance",
    8970: "Barber Shops",
    8980: "Beauty Salons",
    8990: "Nail Salons And Other Personal Care Services",
    9070: "Drycleaning And Laundry Services",
    9080: "Death Care Services",
    9090: "Other Personal Services",
    9160: "Religious Organizations",
    9170: "Civic, Social, Advocacy Organizations, And Grantmaking And Giving Services",
    9180: "Labor Unions And Similar Labor Organizations",
    9190: "Business, Professional, Political, And Similar Organizations",
    9290: "Private Households",
    9370: "Executive Offices And Legislative Bodies",
    9380: "Public Finance Activities",
    9390: "Other General Government And Support",
    9470: "Justice, Public Order, And Safety Activities",
    9480: "Administration Of Human Resource Programs",
    9490: "Administration Of Environmental Quality, And Housing Programs, Urban Planning And Community Development",
    9570: "Administration Of Economic Programs And Space Research And Technology",
    9590: "National Security And International Affairs",
    9670: "U.S. Army",
    9680: "U.S. Air Force",
    9690: "U.S. Navy",
    9770: "U.S. Marines",
    9780: "U.S. Coast Guard",
    9790: "Armed Forces, Branch Not Specified",
    9870: "Military Reserves Or National Guard",
    9920: "Unemployed, Last Worked 5 Years Ago Or Earlier Or Never Worked",
}

# POBP - Place of birth (high-cardinality). US births reuse the state FIPS map;
# a few common foreign birthplaces are labeled too.
POBP_LABELS = {code: f"{name} (USA)" for code, name in STATE_FIPS.items()}
POBP_LABELS.update({
    207: "China", 210: "India", 215: "Philippines", 303: "Mexico",
})


# Continuous fields: (label, from, to, default)
CONTINUOUS_FIELDS = {
    "AGEP": ("Age (years)", 18, 64, 41),
    "WKHP": ("Usual hours worked / week", 1, 99, 40),
    "WKWN": ("Weeks worked in past year", 1, 52, 52),
}

# Low-cardinality categoricals: (label, {code: label}, default_code)
LOWCARD_FIELDS = {
    "COW":      ("Class of worker", COW_LABELS, 1),
    "SCHL":     ("Education", SCHL_LABELS, 21),
    "MAR":      ("Marital status", MAR_LABELS, 1),
    "RELSHIPP": ("Relationship to householder", RELSHIPP_LABELS, 20),
    "SEX":      ("Sex", SEX_LABELS, 1),
    "RAC1P":    ("Race", RAC1P_LABELS, 1),
    "HISP":     ("Hispanic origin", HISP_LABELS, 1),
    "CIT":      ("Citizenship", CIT_LABELS, 1),
    "NATIVITY": ("Nativity", NATIVITY_LABELS, 1),
    "ENG":      ("English ability", ENG_LABELS, 0),
    "DIS":      ("Disability", DIS_LABELS, 2),
    "ST":       ("State of residence", ST_LABELS, 6),
}

# High-cardinality categoricals: (label, {code: label}, default_code)
HIGHCARD_FIELDS = {
    "OCCP": ("Occupation (OCCP code)", OCCP_LABELS, 4720),
    "INDP": ("Industry (INDP code)", INDP_LABELS, 8680),
    "POBP": ("Place of birth (POBP code)", POBP_LABELS, 6),
}


# =============================================================================
# Feature engineering -- must reproduce the exp4 notebook exactly.
# =============================================================================
def preprocess(raw: dict, bundle: dict) -> pd.DataFrame:
    """Turn the 18 raw GUI inputs into the model's 32-column feature frame."""
    te = bundle["target_enc_maps"]
    fq = bundle["freq_maps"]
    gm = bundle["global_mean"]
    cat = bundle["cat_features"]
    feat_cols = bundle["features"]

    r = dict(raw)
    # Engineered interaction / Mincer terms
    r["ANNUAL_HOURS"] = r["WKHP"] * r["WKWN"]
    r["AGE2"]         = r["AGEP"] ** 2
    r["SCHLxAGEP"]    = r["SCHL"] * r["AGEP"]

    # Leak-safe target + frequency encodings for the 4 high-card codes
    for c in ("OCCP", "INDP", "POBP", "ST"):
        r[c + "_te"]   = float(te[c].get(r[c], gm))
        r[c + "_freq"] = float(fq[c].get(r[c], 0.0))

    # OCCP x INDP pair target encoding (same key as training)
    pair_key = int(r["OCCP"]) * 100000 + int(r["INDP"])
    r["OCCPINDP_te"] = float(te["_OCCP_INDP"].get(pair_key, gm))

    # Expected occupation/industry pay x labour supply
    r["OCCPte_x_HOURS"] = r["OCCP_te"] * r["ANNUAL_HOURS"]
    r["INDPte_x_HOURS"] = r["INDP_te"] * r["ANNUAL_HOURS"]

    df = pd.DataFrame([r])
    for c in cat:
        df[c] = df[c].astype(int)
    return df.reindex(columns=feat_cols)   # exact columns, exact order


# =============================================================================
# Small UI helpers
# =============================================================================
def _options_to_display(labels: dict, codes):
    """Build 'code - label' strings (label optional) and a parallel code list."""
    display, code_list = [], []
    for code in codes:
        lbl = labels.get(code)
        display.append(f"{code} - {lbl}" if lbl else f"{code}")
        code_list.append(code)
    return display, code_list


def _attach_autowidth_popdown(combo, cap=80):
    """Widen a combobox's drop-down list to fit its longest value, so long OCCP/INDP
    titles aren't clipped. The entry box keeps its compact size; only the popup grows."""
    def _post():
        values = combo.cget("values")
        if not values:
            return
        longest = max(len(str(v)) for v in values)
        try:
            popdown = combo.tk.call("ttk::combobox::PopdownWindow", combo)
            combo.tk.call(f"{popdown}.f.l", "configure", "-width", min(longest + 1, cap))
        except tk.TclError:
            pass
    combo.configure(postcommand=_post)


# =============================================================================
# Application
# =============================================================================
class App(tk.Tk):
    PAD = dict(padx=8, pady=4)

    def __init__(self):
        super().__init__()
        self.title("Project Echo - Income Predictor (Lite)")
        self.resizable(True, True)
        self.bundle = None
        self.widgets = {}      # field -> (widget, code_list or None)
        self._load_model()
        self._build_ui()

    # ---- model loading (never crash on a bad/missing model) -----------------
    def _load_model(self):
        try:
            self.bundle = joblib.load(MODEL_PATH)
            # rebuild the category dtypes XGBoost/LightGBM need at predict time
            self.cat_dtypes = {
                c: CategoricalDtype(categories=cats)
                for c, cats in self.bundle["cat_categories"].items()
            }
        except Exception as e:
            self.bundle = None
            messagebox.showwarning(
                "Model not loaded",
                f"Could not load model.pkl:\n{e}\n\n"
                "The form still opens, but predictions are disabled.\n"
                f"Expected file: {MODEL_PATH}")

    # ---- UI construction ----------------------------------------------------
    def _build_ui(self):
        # Header banner
        header = tk.Frame(self, bg="#1f3a5f")
        header.grid(row=0, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)            # let the rows stretch when the window is widened
        tk.Label(header, text="Project Echo  -  Socio-Economic Income Predictor",
                 bg="#1f3a5f", fg="white",
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=16, pady=(10, 2))
        tk.Label(header, text="2024 ACS PUMS  -  XGBoost + LightGBM + CatBoost blend (Experiment 4)",
                 bg="#1f3a5f", fg="#bcd2ee",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=16, pady=(0, 10))

        # Form (two columns of fields)
        form = tk.LabelFrame(self, text="  Person attributes  ", font=("Segoe UI", 10, "bold"),
                             padx=10, pady=8)
        form.grid(row=1, column=0, sticky="ew", padx=12, pady=10)
        form.columnconfigure(1, weight=1)            # the two widget columns grow on resize
        form.columnconfigure(4, weight=1)

        # Assemble the field order: continuous, then low-card, then high-card.
        ordered = (list(CONTINUOUS_FIELDS) + list(LOWCARD_FIELDS) + list(HIGHCARD_FIELDS))
        half = (len(ordered) + 1) // 2
        for idx, field in enumerate(ordered):
            col_block = 0 if idx < half else 1
            row = idx if idx < half else idx - half
            self._add_field(form, field, row, col_block * 3)

        # Buttons
        btns = tk.Frame(self)
        btns.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        ttk.Button(btns, text="Predict income", command=self._predict).pack(side="left", padx=4)
        ttk.Button(btns, text="Reset", command=self._reset).pack(side="left", padx=4)

        # Result panel
        res = tk.LabelFrame(self, text="  Prediction  ", font=("Segoe UI", 10, "bold"),
                            padx=10, pady=8)
        res.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        res.columnconfigure(0, weight=1)
        self.result_var = tk.StringVar(value="--")
        self.detail_var = tk.StringVar(value="Enter attributes and click Predict.")
        self.result_lbl = tk.Label(res, textvariable=self.result_var,
                                    font=("Segoe UI", 22, "bold"), fg="#1f3a5f")
        self.result_lbl.grid(row=0, column=0, sticky="w")
        tk.Label(res, textvariable=self.detail_var, font=("Segoe UI", 9),
                 fg="#444", justify="left").grid(row=1, column=0, sticky="w")

    def _add_field(self, parent, field, row, col):
        """Create one labeled widget for a field and register it."""
        if field in CONTINUOUS_FIELDS:
            label, lo, hi, default = CONTINUOUS_FIELDS[field]
            var = tk.StringVar(value=str(default))
            w = ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=22)
            self.widgets[field] = (w, None)
        elif field in LOWCARD_FIELDS:
            label, labels, default = LOWCARD_FIELDS[field]
            codes = sorted(labels)
            display, code_list = _options_to_display(labels, codes)
            w = ttk.Combobox(parent, values=display, state="readonly", width=30)
            w.current(code_list.index(default) if default in code_list else 0)
            self.widgets[field] = (w, code_list)
        else:  # high-cardinality -> editable combobox of all valid codes
            label, labels, default = HIGHCARD_FIELDS[field]
            codes = self._valid_codes(field)
            display, code_list = _options_to_display(labels, codes)
            w = ttk.Combobox(parent, values=display, state="normal", width=30)
            if default in code_list:
                w.current(code_list.index(default))
            elif code_list:
                w.current(0)
            self.widgets[field] = (w, code_list)

        tk.Label(parent, text=label, anchor="w").grid(row=row, column=col, sticky="w", **self.PAD)
        w.grid(row=row, column=col + 1, sticky="ew", **self.PAD)
        if isinstance(w, ttk.Combobox):
            _attach_autowidth_popdown(w)

    def _valid_codes(self, field):
        """Valid codes for a high-card field, taken from the model's own maps."""
        if self.bundle is not None:
            try:
                return sorted(int(v) for v in self.bundle["freq_maps"][field].index)
            except Exception:
                pass
        return sorted(HIGHCARD_FIELDS[field][1])   # fallback: just the labeled ones

    # ---- input collection + validation --------------------------------------
    def _collect_raw(self) -> dict:
        raw = {}
        # continuous
        for field, (label, lo, hi, _d) in CONTINUOUS_FIELDS.items():
            w, _ = self.widgets[field]
            try:
                val = int(float(w.get()))
            except ValueError:
                raise ValueError(f"'{label}' must be a whole number.")
            if not (lo <= val <= hi):
                raise ValueError(f"'{label}' must be between {lo} and {hi}.")
            raw[field] = val
        # low-card -> map combobox index to code
        for field, (label, _labels, _d) in LOWCARD_FIELDS.items():
            w, code_list = self.widgets[field]
            raw[field] = code_list[w.current()]
        # high-card -> parse leading integer from selection or free text
        for field, (label, _labels, _d) in HIGHCARD_FIELDS.items():
            w, _code_list = self.widgets[field]
            text = w.get().strip()
            try:
                raw[field] = int(text.split("-")[0].strip())
            except ValueError:
                raise ValueError(f"'{label}' must be a numeric ACS code.")
        return raw

    # ---- prediction ---------------------------------------------------------
    def _predict(self):
        if self.bundle is None:
            messagebox.showerror("No model", "model.pkl is not loaded; cannot predict.")
            return
        try:
            raw = self._collect_raw()
            X = preprocess(raw, self.bundle)
            Xc = X.copy()
            for c, dt in self.cat_dtypes.items():
                Xc[c] = Xc[c].astype(dt)

            scale = self.bundle["wagp_to_dollars"]
            models = self.bundle["models"]
            # XGBoost & LightGBM need category dtype; CatBoost takes the int frame.
            log_preds = {
                "XGBoost":  float(models["XGBoost"].predict(Xc)[0]),
                "LightGBM": float(models["LightGBM"].predict(Xc)[0]),
                "CatBoost": float(models["CatBoost"].predict(X)[0]),
            }
            dollars = {n: float(np.expm1(lp) * scale) for n, lp in log_preds.items()}
            blend = float(np.expm1(np.mean(list(log_preds.values()))) * scale)
        except ValueError as ve:
            messagebox.showerror("Invalid input", str(ve))
            return
        except Exception as e:
            messagebox.showerror("Prediction failed", f"{type(e).__name__}: {e}")
            return

        band, color = self._band(blend)
        self.result_var.set(f"${blend:,.0f} / year")
        self.result_lbl.config(fg=color)
        breakdown = "   ".join(f"{n} ${v/1000:,.1f}k" for n, v in dollars.items())
        self.detail_var.set(
            f"{band}\nPer-model:  {breakdown}\n"
            f"(3-booster blend of exp4; predicted annual wage/salary income, WAGP)")

    @staticmethod
    def _band(amount):
        if amount < 25_000:
            return "Lower income band", "#c0392b"
        if amount < 50_000:
            return "Lower-middle income band", "#e67e22"
        if amount < 80_000:
            return "Middle income band", "#2980b9"
        if amount < 120_000:
            return "Upper-middle income band", "#16a085"
        return "High income band", "#27ae60"

    # ---- reset --------------------------------------------------------------
    def _reset(self):
        for field, (w, code_list) in self.widgets.items():
            if field in CONTINUOUS_FIELDS:
                w.delete(0, "end")
                w.insert(0, str(CONTINUOUS_FIELDS[field][3]))
            elif field in LOWCARD_FIELDS:
                default = LOWCARD_FIELDS[field][2]
                w.current(code_list.index(default) if default in code_list else 0)
            else:
                default = HIGHCARD_FIELDS[field][2]
                w.set(f"{default} - {HIGHCARD_FIELDS[field][1].get(default, '')}".rstrip(" -")
                      if default in code_list else (w["values"][0] if w["values"] else ""))
        self.result_var.set("--")
        self.result_lbl.config(fg="#1f3a5f")
        self.detail_var.set("Enter attributes and click Predict.")


if __name__ == "__main__":
    App().mainloop()
