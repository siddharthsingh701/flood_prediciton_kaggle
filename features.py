
import numpy as np
import pandas as pd
import config as config

def create_features_1(data):
    df = data.copy()
    cols = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
       'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
       'Siltation', 'AgriculturalPractices', 'Encroachments',
       'IneffectiveDisasterPreparedness', 'DrainageSystems',
       'CoastalVulnerability', 'Landslides', 'Watersheds',
       'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
       'InadequatePlanning', 'PoliticalFactors']
    df['ClimateImpact'] = df.MonsoonIntensity+df.ClimateChange
    df['AnthropogenicPressure'] = df.Deforestation+df.Urbanization+df.AgriculturalPractices+df.Encroachments
    df['InfrastructureQuality'] = df.DamsQuality+df.DrainageSystems+df.DeterioratingInfrastructure
    df['CoastalVulnerabilityTotal'] = df.CoastalVulnerability+df.Landslides
    df['PreventiveMeasuresEfficiency'] = df.RiverManagement+df.IneffectiveDisasterPreparedness+df.InadequatePlanning
    df['EcosystemImpact'] = df.WetlandLoss+df.Watersheds
    df['SocioPoliticalContext'] = df.PopulationScore+df.PoliticalFactors
    df['Land_Use_Pressure'] = df.Urbanization+df.Deforestation+df.AgriculturalPractices
    df['Environmental_Degradation']=df.Deforestation+df.Siltation+df.WetlandLoss+df.Landslides
    df['Infrastructure_Vulnerability'] = df.DeterioratingInfrastructure+df.InadequatePlanning
    df['Community_Preparednessg']= df.IneffectiveDisasterPreparedness+df.PoliticalFactors
    df['Population_Density_Vulnerable_Areas']= df.PopulationScore+df.CoastalVulnerability
    df['Climate_Change_Impact']= df.ClimateChange+df.MonsoonIntensity
    df['River_Health']= df.RiverManagement+df.DamsQuality
    df['ClimateImpact'] = df.MonsoonIntensity+df.ClimateChange
    df['AnthropogenicPressure'] = df.Deforestation+df.Urbanization+df.AgriculturalPractices+df.Encroachments
    df['InfrastructureQuality'] = df.DamsQuality+df.DrainageSystems+df.DeterioratingInfrastructure
    df['CoastalVulnerabilityTotal'] = df.CoastalVulnerability+df.Landslides
    df['PreventiveMeasuresEfficiency'] = df.RiverManagement+df.IneffectiveDisasterPreparedness+df.InadequatePlanning
    df['EcosystemImpact'] = df.WetlandLoss+df.Watersheds
    df['SocioPoliticalContext'] = df.PopulationScore+df.PoliticalFactors
    df['Land_Use_Pressure']=df.Urbanization+df.Deforestation+df.AgriculturalPractices
    df['Environmental_Degradation']=df.Deforestation+df.Siltation+df.WetlandLoss+df.Landslides
    df['Infrastructure_Vulnerability'] = df.DeterioratingInfrastructure+df.InadequatePlanning
    df['Community_Preparednessg']= df.IneffectiveDisasterPreparedness+df.PoliticalFactors
    df['Population_Density_Vulnerable_Areas']= df.PopulationScore+df.CoastalVulnerability
    df['Climate_Change_Impact']= df.ClimateChange+df.MonsoonIntensity
    df['River_Health']= df.RiverManagement+df.DamsQuality
    df['sum']= df.sum(axis=1)
    df['mean']= df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    df['var'] = df.var(axis=1)
    df['skew'] = df.skew(axis=1)
    df['kurt'] = df.kurt(axis=1)
    df['meadian'] = df.median(axis=1)
    df['quant_25'] = df.quantile(0.25,axis=1)
    df['quant_75'] = df.quantile(0.75,axis=1)
    df['sum>72'] = np.where(df['sum']>72,1,0)
    df['sum>100'] = np.where(df['sum']>100,1,0)
    df['sum>50'] = np.where(df['sum']>50,1,0)
    df['range']= df['max']-df['min']
    for col in cols:
        df[f"{col}_2"]= df[col]**2
        df[f"{col}_3"]= df[col]**3
        df[f"{col}_3"]= df[col]**4
    for col in cols:
        if col not in ['id','FloodProbability']:
            df[f"mad_{col}"] = df[col] - df[col].median()
            df[f"mean_{col}"] = df[col] - df[col].mean()
            df[f"std_{col}"] = df[col] - df[col].std()
    return df

def create_features_2(data):
    df = data.copy()
    cols = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
       'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
       'Siltation', 'AgriculturalPractices', 'Encroachments',
       'IneffectiveDisasterPreparedness', 'DrainageSystems',
       'CoastalVulnerability', 'Landslides', 'Watersheds',
       'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
       'InadequatePlanning', 'PoliticalFactors']
    df['ClimateImpact'] = df.MonsoonIntensity+df.ClimateChange
    df['AnthropogenicPressure'] = df.Deforestation+df.Urbanization+df.AgriculturalPractices+df.Encroachments
    df['InfrastructureQuality'] = df.DamsQuality+df.DrainageSystems+df.DeterioratingInfrastructure
    df['CoastalVulnerabilityTotal'] = df.CoastalVulnerability+df.Landslides
    df['PreventiveMeasuresEfficiency'] = df.RiverManagement+df.IneffectiveDisasterPreparedness+df.InadequatePlanning
    df['EcosystemImpact'] = df.WetlandLoss+df.Watersheds
    df['SocioPoliticalContext'] = df.PopulationScore+df.PoliticalFactors
    df['Land_Use_Pressure'] = df.Urbanization+df.Deforestation+df.AgriculturalPractices
    df['Environmental_Degradation']=df.Deforestation+df.Siltation+df.WetlandLoss+df.Landslides
    df['Infrastructure_Vulnerability'] = df.DeterioratingInfrastructure+df.InadequatePlanning
    df['Community_Preparednessg']= df.IneffectiveDisasterPreparedness+df.PoliticalFactors
    df['Population_Density_Vulnerable_Areas']= df.PopulationScore+df.CoastalVulnerability
    df['Climate_Change_Impact']= df.ClimateChange+df.MonsoonIntensity
    df['River_Health']= df.RiverManagement+df.DamsQuality
    df['ClimateImpact'] = df.MonsoonIntensity+df.ClimateChange
    df['AnthropogenicPressure'] = df.Deforestation+df.Urbanization+df.AgriculturalPractices+df.Encroachments
    df['InfrastructureQuality'] = df.DamsQuality+df.DrainageSystems+df.DeterioratingInfrastructure
    df['CoastalVulnerabilityTotal'] = df.CoastalVulnerability+df.Landslides
    df['PreventiveMeasuresEfficiency'] = df.RiverManagement+df.IneffectiveDisasterPreparedness+df.InadequatePlanning
    df['EcosystemImpact'] = df.WetlandLoss+df.Watersheds
    df['SocioPoliticalContext'] = df.PopulationScore+df.PoliticalFactors
    df['Land_Use_Pressure']=df.Urbanization+df.Deforestation+df.AgriculturalPractices
    df['Environmental_Degradation']=df.Deforestation+df.Siltation+df.WetlandLoss+df.Landslides
    df['Infrastructure_Vulnerability'] = df.DeterioratingInfrastructure+df.InadequatePlanning
    df['Community_Preparednessg']= df.IneffectiveDisasterPreparedness+df.PoliticalFactors
    df['Population_Density_Vulnerable_Areas']= df.PopulationScore+df.CoastalVulnerability
    df['Climate_Change_Impact']= df.ClimateChange+df.MonsoonIntensity
    df['River_Health']= df.RiverManagement+df.DamsQuality
    df['sum']= df.sum(axis=1)
    df['mean']= df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    df['var'] = df.var(axis=1)
    df['skew'] = df.skew(axis=1)
    df['kurt'] = df.kurt(axis=1)
    df['meadian'] = df.median(axis=1)
    df['quant_25'] = df.quantile(0.25,axis=1)
    df['quant_75'] = df.quantile(0.75,axis=1)
    df['sum>72'] = np.where(df['sum']>72,1,0)
    df['sum>100'] = np.where(df['sum']>100,1,0)
    df['sum>50'] = np.where(df['sum']>50,1,0)
    df['range']= df['max']-df['min']
    for col in cols:
        df[f"{col}_2"]= df[col]**2
        df[f"{col}_3"]= df[col]**3
        df[f"{col}_3"]= df[col]**4
        # Log Features
        df[f"log_{col}"] = np.log1p(df[col]+1e-4)  
    for col in cols:
        if col not in ['id','FloodProbability']:
            df[f"mad_{col}"] = df[col] - df[col].median()
            df[f"mean_{col}"] = df[col] - df[col].mean()
            df[f"std_{col}"] = df[col] - df[col].std()
    return df

def create_features_3(data):
    df = data.copy()
    cols = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
       'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
       'Siltation', 'AgriculturalPractices', 'Encroachments',
       'IneffectiveDisasterPreparedness', 'DrainageSystems',
       'CoastalVulnerability', 'Landslides', 'Watersheds',
       'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
       'InadequatePlanning', 'PoliticalFactors']
    df['ClimateImpact'] = df.MonsoonIntensity+df.ClimateChange
    df['AnthropogenicPressure'] = df.Deforestation+df.Urbanization+df.AgriculturalPractices+df.Encroachments
    df['InfrastructureQuality'] = df.DamsQuality+df.DrainageSystems+df.DeterioratingInfrastructure
    df['CoastalVulnerabilityTotal'] = df.CoastalVulnerability+df.Landslides
    df['PreventiveMeasuresEfficiency'] = df.RiverManagement+df.IneffectiveDisasterPreparedness+df.InadequatePlanning
    df['EcosystemImpact'] = df.WetlandLoss+df.Watersheds
    df['SocioPoliticalContext'] = df.PopulationScore+df.PoliticalFactors
    df['Land_Use_Pressure'] = df.Urbanization+df.Deforestation+df.AgriculturalPractices
    df['Environmental_Degradation']=df.Deforestation+df.Siltation+df.WetlandLoss+df.Landslides
    df['Infrastructure_Vulnerability'] = df.DeterioratingInfrastructure+df.InadequatePlanning
    df['Community_Preparednessg']= df.IneffectiveDisasterPreparedness+df.PoliticalFactors
    df['Population_Density_Vulnerable_Areas']= df.PopulationScore+df.CoastalVulnerability
    df['Climate_Change_Impact']= df.ClimateChange+df.MonsoonIntensity
    df['River_Health']= df.RiverManagement+df.DamsQuality
    df['ClimateImpact'] = df.MonsoonIntensity+df.ClimateChange
    df['AnthropogenicPressure'] = df.Deforestation+df.Urbanization+df.AgriculturalPractices+df.Encroachments
    df['InfrastructureQuality'] = df.DamsQuality+df.DrainageSystems+df.DeterioratingInfrastructure
    df['CoastalVulnerabilityTotal'] = df.CoastalVulnerability+df.Landslides
    df['PreventiveMeasuresEfficiency'] = df.RiverManagement+df.IneffectiveDisasterPreparedness+df.InadequatePlanning
    df['EcosystemImpact'] = df.WetlandLoss+df.Watersheds
    df['SocioPoliticalContext'] = df.PopulationScore+df.PoliticalFactors
    df['Land_Use_Pressure']=df.Urbanization+df.Deforestation+df.AgriculturalPractices
    df['Environmental_Degradation']=df.Deforestation+df.Siltation+df.WetlandLoss+df.Landslides
    df['Infrastructure_Vulnerability'] = df.DeterioratingInfrastructure+df.InadequatePlanning
    df['Community_Preparednessg']= df.IneffectiveDisasterPreparedness+df.PoliticalFactors
    df['Population_Density_Vulnerable_Areas']= df.PopulationScore+df.CoastalVulnerability
    df['Climate_Change_Impact']= df.ClimateChange+df.MonsoonIntensity
    df['River_Health']= df.RiverManagement+df.DamsQuality
    df['sum']= df.sum(axis=1) # for tree models 
    df['product']= df.product(axis=1)
    df['special1'] = df['sum'].isin(np.arange(72, 76)) # for linear models
    df['special2'] = df['product'].isin(np.arange(72, 76)) 
    df['mean']= df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    df['var'] = df.var(axis=1)
    df['skew'] = df.skew(axis=1)
    df['kurt'] = df.kurt(axis=1)
    df['meadian'] = df.median(axis=1)
    df['quant_25'] = df.quantile(0.25,axis=1)
    df['quant_75'] = df.quantile(0.75,axis=1)
    df['sum>72'] = np.where(df['sum']>72,1,0)
    df['sum>100'] = np.where(df['sum']>100,1,0)
    df['sum>50'] = np.where(df['sum']>50,1,0)
    df['range']= df['max']-df['min']
    for col in cols:
        df[f"{col}_2"]= df[col]**2
        df[f"{col}_3"]= df[col]**3
        df[f"{col}_3"]= df[col]**4
        # Log Features
        df[f"log_{col}"] = np.log1p(df[col]+1e-4)
        df[f"log_{col}"] = np.log2(df[col]+1e-4) 

    for col in cols:
        if col not in ['id','FloodProbability']:
            df[f"mad_{col}"] = df[col] - df[col].median()
            df[f"mean_{col}"] = df[col] - df[col].mean()
            df[f"std_{col}"] = df[col] - df[col].std()
    return df

def create_features_4(data):
    df = data.copy()

    df['fsum'] = df[config.INITIAL_FEATURES].sum(axis=1) # for tree models
    df['special1'] = df['fsum'].isin(np.arange(72, 76)) # for linear models

    log_features = [f"log_{col}" for col in config.INITIAL_FEATURES]
    log2_features = [f"log_{col}" for col in config.INITIAL_FEATURES]

    exp_features = [f"exp_{col}" for col in config.INITIAL_FEATURES]
    exp2_features = [f"exp2_{col}" for col in config.INITIAL_FEATURES]
    exp3_features = [f"exp3_{col}" for col in config.INITIAL_FEATURES]
    exp4_features = [f"exp4_{col}" for col in config.INITIAL_FEATURES]
    new_cols = []

    df['fsum2'] = df[config.INITIAL_FEATURES].product(axis=1)
    df['zero_count'] = (df[config.INITIAL_FEATURES] < 10).sum(axis=1)
    df['one_count'] = (df[config.INITIAL_FEATURES] > 10).sum(axis=1)
    
    df['special2'] = df['fsum2'].isin(np.arange(72, 76)) 

    for col in config.INITIAL_FEATURES:
        df[f"log_{col}"] = np.log1p(df[col]+1e-4)  
    df['log_sum'] = df[log_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"log2_{col}"] = np.log2(df[col]+1e-4)  
    df['log2_sum'] = df[log2_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp_{col}"] = 1.2**(df[col])

    df['exp_sum'] = df[exp_features].sum(axis=1)
    df['exp_prod'] = df[exp_features].product(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp2_{col}"] = np.exp(df[col])
    df['exp2_sum'] = df[exp2_features].sum(axis=1)


    for col in config.INITIAL_FEATURES:
        df[f"exp3_{col}"] = 4**(df[col])
    df['exp3_sum'] = df[exp3_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp4_{col}"] = 6**(df[col])
    df['exp4_sum'] = df[exp4_features].sum(axis=1)

    feats = list(config.INITIAL_FEATURES)+['fsum','one_count','fsum2','exp_sum','log_sum','log2_sum','exp2_sum','exp3_sum']
    df = df[feats]
    return df 

def create_features_5(data):
    df = data.copy()

    df['fsum'] = df[config.INITIAL_FEATURES].sum(axis=1) # for tree models
    df['special1'] = df['fsum'].isin(np.arange(72, 76)) # for linear models
    df['special1'] = np.where(df['special1']==True,1,0)

    log_features = [f"log_{col}" for col in config.INITIAL_FEATURES]
    log2_features = [f"log_{col}" for col in config.INITIAL_FEATURES]

    exp_features = [f"exp_{col}" for col in config.INITIAL_FEATURES]
    exp2_features = [f"exp2_{col}" for col in config.INITIAL_FEATURES]
    exp3_features = [f"exp3_{col}" for col in config.INITIAL_FEATURES]
    exp4_features = [f"exp4_{col}" for col in config.INITIAL_FEATURES]
    new_cols = []

    df['fsum2'] = df[config.INITIAL_FEATURES].product(axis=1)
    df['zero_count'] = (df[config.INITIAL_FEATURES] < 10).sum(axis=1)
    df['one_count'] = (df[config.INITIAL_FEATURES] > 10).sum(axis=1)
    
    df['special2'] = df['fsum2'].isin(np.arange(72, 76)) 
    df['special2'] = np.where(df['special2']==True,1,0)
    for col in config.INITIAL_FEATURES:
        df[f"log_{col}"] = np.log1p(df[col]+1e-4)  
    df['log_sum'] = df[log_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"log2_{col}"] = np.log2(df[col]+1e-4)  
    df['log2_sum'] = df[log2_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp_{col}"] = 1.2**(df[col])

    df['exp_sum'] = df[exp_features].sum(axis=1)
    df['exp_prod'] = df[exp_features].product(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp2_{col}"] = np.exp(df[col])
    df['exp2_sum'] = df[exp2_features].sum(axis=1)


    for col in config.INITIAL_FEATURES:
        df[f"exp3_{col}"] = 4**(df[col])
    df['exp3_sum'] = df[exp3_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp4_{col}"] = 6**(df[col])
    df['exp4_sum'] = df[exp4_features].sum(axis=1)

    feats = list(config.INITIAL_FEATURES)+['fsum','one_count','fsum2','exp_sum','log_sum','log2_sum','exp2_sum','exp3_sum']+['special1','special2']
    df = df[feats]
    return df 

def create_features_6(data):
    df = data.copy()

    df['fsum'] = df[config.INITIAL_FEATURES].sum(axis=1) # for tree models
    df['special1'] = df['fsum'].isin(np.arange(72, 76)) # for linear models
    df['special1'] = np.where(df['special1']==True,1,0)

    log_features = [f"log_{col}" for col in config.INITIAL_FEATURES]
    log2_features = [f"log_{col}" for col in config.INITIAL_FEATURES]

    exp_features = [f"exp_{col}" for col in config.INITIAL_FEATURES]
    exp2_features = [f"exp2_{col}" for col in config.INITIAL_FEATURES]
    exp3_features = [f"exp3_{col}" for col in config.INITIAL_FEATURES]
    exp4_features = [f"exp4_{col}" for col in config.INITIAL_FEATURES]
    new_cols = []

    df['fsum2'] = df[config.INITIAL_FEATURES].product(axis=1)
    df['zero_count'] = (df[config.INITIAL_FEATURES] < 10).sum(axis=1)
    df['one_count'] = (df[config.INITIAL_FEATURES] > 10).sum(axis=1)
    
    df['special2'] = df['fsum2'].isin(np.arange(72, 76)) 
    df['special2'] = np.where(df['special2']==True,1,0)
    for col in config.INITIAL_FEATURES:
        df[f"log_{col}"] = np.log1p(df[col]+1e-4)  
    df['log_sum'] = df[log_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"log2_{col}"] = np.log2(df[col]+1e-4)  
    df['log2_sum'] = df[log2_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp_{col}"] = 1.2**(df[col])

    df['exp_sum'] = df[exp_features].sum(axis=1)
    df['exp_prod'] = df[exp_features].product(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp2_{col}"] = np.exp(df[col])
    df['exp2_sum'] = df[exp2_features].sum(axis=1)


    for col in config.INITIAL_FEATURES:
        df[f"exp3_{col}"] = 4**(df[col])
    df['exp3_sum'] = df[exp3_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp4_{col}"] = 6**(df[col])
    df['exp4_sum'] = df[exp4_features].sum(axis=1)

    feats = list(config.INITIAL_FEATURES)+['fsum','one_count','fsum2','exp_sum','log_sum','log2_sum','exp2_sum','exp3_sum']+log_features
    df = df[feats]
    return df 

def create_features_7(data):
    df = data.copy()

    df['fsum'] = df[config.INITIAL_FEATURES].sum(axis=1) # for tree models
    df['special1'] = df['fsum'].isin(np.arange(72, 76)) # for linear models
    df['special1'] = np.where(df['special1']==True,1,0)

    log_features = [f"log_{col}" for col in config.INITIAL_FEATURES]
    log2_features = [f"log_{col}" for col in config.INITIAL_FEATURES]

    exp_features = [f"exp_{col}" for col in config.INITIAL_FEATURES]
    exp2_features = [f"exp2_{col}" for col in config.INITIAL_FEATURES]
    exp3_features = [f"exp3_{col}" for col in config.INITIAL_FEATURES]
    exp4_features = [f"exp4_{col}" for col in config.INITIAL_FEATURES]
    new_cols = []

    df['fsum2'] = df[config.INITIAL_FEATURES].product(axis=1)
    df['zero_count'] = (df[config.INITIAL_FEATURES] < 10).sum(axis=1)
    df['one_count'] = (df[config.INITIAL_FEATURES] > 10).sum(axis=1)
    
    df['special2'] = df['fsum2'].isin(np.arange(72, 76)) 
    df['special2'] = np.where(df['special2']==True,1,0)
    for col in config.INITIAL_FEATURES:
        df[f"log_{col}"] = np.log1p(df[col]+1e-4)  
    df['log_sum'] = df[log_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"log2_{col}"] = np.log2(df[col]+1e-4)  
    df['log2_sum'] = df[log2_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp_{col}"] = 1.2**(df[col])

    df['exp_sum'] = df[exp_features].sum(axis=1)
    df['exp_prod'] = df[exp_features].product(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp2_{col}"] = np.exp(df[col])
    df['exp2_sum'] = df[exp2_features].sum(axis=1)


    for col in config.INITIAL_FEATURES:
        df[f"exp3_{col}"] = 4**(df[col])
    df['exp3_sum'] = df[exp3_features].sum(axis=1)

    for col in config.INITIAL_FEATURES:
        df[f"exp4_{col}"] = 6**(df[col])
    df['exp4_sum'] = df[exp4_features].sum(axis=1)

    feats = list(config.INITIAL_FEATURES)+['fsum','fsum2','exp_sum','log_sum','exp2_sum','exp3_sum']
    df = df[feats]
    return df