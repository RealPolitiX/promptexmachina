
##### MMLU dataset #####

mmlu = {}
mmlu['topics'] = ['abstract_algebra',
                'anatomy',
                'astronomy',
                'business_ethics',
                'clinical_knowledge',
                'college_biology',
                'college_chemistry',
                'college_computer_science',
                'college_mathematics',
                'college_medicine',
                'college_physics',
                'computer_security',
                'conceptual_physics',
                'econometrics',
                'electrical_engineering',
                'elementary_mathematics',
                'formal_logic',
                'global_facts',
                'high_school_biology',
                'high_school_chemistry',
                'high_school_computer_science',
                'high_school_european_history',
                'high_school_geography',
                'high_school_government_and_politics',
                'high_school_macroeconomics',
                'high_school_mathematics',
                'high_school_microeconomics',
                'high_school_physics',
                'high_school_psychology',
                'high_school_statistics',
                'high_school_us_history',
                'high_school_world_history',
                'human_aging',
                'human_sexuality',
                'international_law',
                'jurisprudence',
                'logical_fallacies',
                'machine_learning',
                'management',
                'marketing',
                'medical_genetics',
                'miscellaneous',
                'moral_disputes',
                'moral_scenarios',
                'nutrition',
                'philosophy',
                'prehistory',
                'professional_accounting',
                'professional_law',
                'professional_medicine',
                'professional_psychology',
                'public_relations',
                'security_studies',
                'sociology',
                'us_foreign_policy',
                'virology',
                'world_religions']


# High schol-level topics
mmlu['high_school'] = [top for top in topics if top.startswith('high_school')]

# College-level topics
mmlu['college'] = [top for top in topics if top.startswith('college')]

# Professional-level topics
mmlu['profs'] = [top for top in topics if top.startswith('professional')]

mmlu['selected'] = ['college_chemistry',
                'college_computer_science',
                'college_medicine',
                'management',
                'moral_disputes',
                'sociology']

##### MedMCQA dataset #####



##### BBQ dataset #####

bbq = {}
bbq['topics'] = ['Age',
                'Disability_status',
                'Gender_identity',
                'Nationality',
                'Physical_appearance',
                'Race_ethnicity',
                'Race_x_SES',
                'Race_x_gender',
                'Religion',
                'SES',
                'Sexual_orientation']

bbq['selected'] = ['Age',
                   'Disability_status',
                   'Physical_appearance',
                   'SES']