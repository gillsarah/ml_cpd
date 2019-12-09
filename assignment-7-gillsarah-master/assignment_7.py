'''
Explore the data with summary statistics and visualiations. You can reuse functions from the relevant earlier assignments for all loading and analysis.
Perform any necessary preprocessing.
Use a test harness to assess which model to use.
Fit a supervised ML model (regression or classification) to the data in that way that lets you make useful predictions.
'''


import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import statsmodels.api as sm
import statsmodels.formula.api as smf 


#Idea: can we predict if a complaint will be sustained based on the catagory of the complaint, 
#the race of the victim, the race of the officer and maybe other demog char?
#or I could see if this predicts if any action taken! make the final_discipline into binary of 
#no action vs all else (if it is SU is very predictive of this, will there be room for other stuff?)

#or I could see if race is predictive of catagory of complaint

'''
may need 
complaints-accused_1967-1999_2016-12.csv.gz
final_finding_desc
'''

#altered from HW 4

#all of the needed data is in my HW7 Repository

path= '/Users/Sarah/Documents/GitHub/assignment-7-gillsarah'

#profile_path = 'unified_data/profiles/officer-profiles.csv.gz' #path for reading from chicago-police-data
profile_path = 'fully-unified-data/profiles/officer-profiles.csv.gz'
codes_path = 'context_data/discipline_penalty_codes.csv'
base_path = 'fully-unified-data/complaints/complaints-{}_2000-2016_2016-11.csv.gz'
file_name = ['accused', 'investigators', 'victims']


def pathmaker(base_path, file):
    return base_path.format(file)



def read_df(path, filename):
    df = pd.read_csv(os.path.join(path, filename))
    return df


def parse_accused(accused_df):
    accused_df.drop(columns = 'row_id', inplace = True)
    final_dummies = pd.get_dummies(accused_df['final_finding'])
    #recommend_dummies = pd.get_dummies(accused_df['recommended_finding'])
    #cite https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
    accused_df['sustained'] = final_dummies['SU']
    #accused_df['recommend_sustain'] = recommend_dummies['SU']
    return accused_df


def parse_investigarots(investigators_df):
    investigators_df.drop(columns = 'row_id', inplace = True)
    investigators_df.rename(columns = {'first_name':'investigator_first_name', 
                                       'last_name':'investigator_last_name',
                                       'middle_initial':'investigator_middle_initial', 
                                       'suffix_name':'investigator_suffix',
                                       'appointed_date': 'date_investigator_appointed', 
                                       'current_star':'investigator_current_star_number',
                                       'current_rank': 'investigator_current_rank', 
                                       'current_unit':'investigator_current_unit'}, inplace = True)
    return investigators_df

def parse_victims(victims_df):
    victims_df.rename(columns = {'gender':'victim_gender', 'age':'victim_age', 
                                 'race':'victim_race'}, inplace = True)
    return victims_df

def parse_profile(profile_df):
    profile_df['org_hire_date'] = pd.to_datetime(profile_df['org_hire_date'], 
                                                 format='%Y-%m-%d')
    #profile_df['birth_year'] = pd.to_datetime(profile_df['birth_year'], format='%Y')
    profile_df['Year_hired'] = profile_df['org_hire_date'].map(lambda d: d.year)
    
    return profile_df

def merge_dfs(dfs):
    '''
    takes a list of dfs, the order is decided in the list dfs. If you change the order, the function
    may need to be tweeked. The suffixes, and merges 3 and 4
    The frist df is accused, the second is investigators, the third is victims, the third is codes
    '''
    merge_0 = dfs[0].merge(dfs[4], how = 'left', on = 'UID', suffixes = ('_accused', '_profile'))
    merge_1 = merge_0.merge(dfs[1], how = 'inner', on =  "cr_id", suffixes = ('_accused','_investigators'))
    merge_2 = merge_1.merge(dfs[2], how = 'inner', on =  "cr_id")
    merge_3 = merge_2.merge(dfs[3], how = 'left', left_on = 'recommended_discipline', right_on = 'CODE')

    merge_3.drop(columns = 'CODE', inplace = True)
    merge_3.rename(columns={'recommended_discipline': 'recommended_discipline_code',
                            'ACTION_TAKEN'          : 'recommended_discipline'      }, inplace = True)       
    
    merge_4 = merge_3.merge(dfs[3], how = 'left', left_on = 'final_discipline', 
                            right_on = 'CODE',suffixes = ('_recommended_discipline', '_final_discipline'))

    merge_4.drop(columns = 'CODE', inplace = True)

    merge_4.rename(columns={'final_discipline': 'final_discipline_code',
                            'ACTION_TAKEN'    : 'final_discipline'     }, inplace = True) 
    merge_4['count'] = 1 
    return merge_4


#proportion sustained: looking at all complaints filed, one entry per accused individual
def total_proportion(accused_df):
    return accused_df['sustained'].sum()/len(accused_df.index)

#proportion sustained of complaints that have a line in victims, investigarots and accused 
#from HW4, but a bit more generalized
def outcome_by_catagory(df, group_by, outcome_word):
    '''
    takes a df, a column to group_by and a dummy colum for an outcome e.g. sustained
    output is a df of just the proportion by each catatory in the group_by col
    '''
    grouped = df.groupby(group_by).sum()
    #grouped[outcome_word]
    grouped['proportion_'+outcome_word] = grouped[outcome_word]/len(df.index)
    df = grouped['proportion_'+outcome_word]

    return df


def complaint_type_outcomes(accused_df, outcome, outcome_word):
    '''
    takes the acccused df, the two letter string for final finding: e.g. 'SU' and a string 
    for the final finding abbreviation meaning (e.g. 'sustained' for 'SU')
    output is a list of the complaint catagories for which the outcome (e.g. 'SU')
    is the most likely final finding
    '''
    crosstab = pd.crosstab(accused_df['final_finding'], accused_df['complaint_category'])
    #cite https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html

    print('The following complaint catagories are most likely to be ' + outcome_word + ':')
    temp_list = []
    for column in crosstab.columns:
        if crosstab[column].idxmax() == outcome:
            print(column)
            temp_list.append(column)
    #cite: https://stackoverflow.com/questions/15741759/find-maximum-value-of-a-column-and-return-the-corresponding-row-values-using-pan
    df = pd.DataFrame(temp_list, columns = ['complaint catagories most likely to be ' + outcome_word]) 
    #cite: https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/
    return df


def export_df(df, path, filename):
    df.to_csv(os.path.join(path, filename))



def main():
    files = []
    for f in file_name:
        files.append(pathmaker(base_path, f))

    #commented out for useing only one data path
    #unzip(data_path, 'unified_data/unified_data.zip', path)

    dfs = []
    for filename in files:
        df = read_df(path, filename)
        if filename.__contains__('accused'):
            dfs.append(parse_accused(df))
        elif filename.__contains__('investigators'):
            dfs.append(parse_investigarots(df))
        elif filename.__contains__('victims'):
            dfs.append(parse_victims(df))
        else:
            print('unexpected file')
    dfs.append(read_df(path, codes_path))
    
    df2 = read_df(path, profile_path)
    dfs.append(parse_profile(df2))

    df = merge_dfs(dfs)

    #proportion = total_proportion(dfs[0])
    #print('Total proportion of complaints that are sustained: {:.4f}'.format(proportion))
    #print(' ')

    #race_df = outcome_by_catagory(df, 'victim_race', 'sustained')


    #outcome_df = complaint_type_outcomes(dfs[0], 'SU', 'sustained')

    export_df(df, path, 'full_df.csv')
    #export_df(race_df, path, 'Proportion of compliants sustained by race.csv')
    #export_df(outcome_df, path, 'Most likely to be sustained.csv')

    return df

df = main()

#df['complaint_category'].unique()
#range(len(df['complaint_category'].unique()))

#complaint_type_outcomes(df, 'NS', '...')
#crosstab = pd.crosstab(df['final_discipline'], df['race'])

#works
def set_id (df, col_name):
    df = df.assign(id=(df[col_name]).astype('category').cat.codes)
    #cite https://stackoverflow.com/questions/45685254/q-pandas-how-to-efficiently-assign-unique-id-to-individuals-with-multiple-ent
    df.rename(columns = {'id': col_name+'_id'}, inplace = True)
    return df 

#call
#id works for scatter, but scatter doens't help enough! I need dummy variables for catagorical data
df = set_id (df, 'victim_race')
df = set_id (df, 'race')
df = set_id (df, 'complaint_category')
df = set_id (df, 'final_finding')


#check
def check_new_col(df, ref_col, new_col, i_list=[10,14,30,55,108]):
    for i in i_list:
        print(df[ref_col][i], df[new_col][i])


def dummy_maker(df, col, new_col_name, value):
    '''
    takes a df, sring col name (col of interest), string new col name (col that will be dummy variable)
    and the value in the col that should be a 1
    returns df with this new col
    '''
    final_dummies = pd.get_dummies(df[col])
    df[new_col_name] = final_dummies[value]
    return df
 
df = dummy_maker(df, 'race', 'white_officer', 'WHITE')

#check
check_new_col(df, 'race', 'white_officer')



col_list = ['final_finding_id','complaint_category_id', 'victim_race_id', 'race_id', 'final_discipline'] 
           # 'current_star', 'current_unit', 'current_rank', 'org_hire_date']
             #'race', 'gender',

def small_df_maker(df, col_list):
    '''
    takes a df and a list of column names in that df, that you want to work with
    '''
    
    drop_list = []
               
    for colname in df.columns:
        if colname in col_list:
            pass
        else:
            drop_list.append(colname)
    df2 = df.drop(columns = drop_list)
    #df3 = dict(tuple(df2.groupby(col1)))
    #cite https://stackoverflow.com/questions/19790790/splitting-dataframe-into-multiple-dataframes
    return df2

df2 = small_df_maker(df, col_list)

sns.pairplot(df2, hue='final_discipline')



#I see some clustering on complaint_category
y = 'birth_year'
x = 'complaint_category_id'
sns.catplot(x=x, y=y, data=df, hue = 'final_finding', alpha = 0.1)
#https://seaborn.pydata.org/tutorial/categorical.html


#ok, this does show that one final finding has more variety of final_discipline
x = 'final_finding_id'
y = 'complaint_category_id'
sns.catplot(x=x, y=y, data=df, hue = 'final_discipline', alpha = 0.1)
#https://seaborn.pydata.org/tutorial/categorical.html

outcome_by_catagory(df, 'complaint_category', 'sustained')
outcome_by_catagory(df, 'race', 'sustained')

'''
#take len of unique entries in the col and yields numbers 1-n
def id_gen(n): 
    num_list = list(range(n))
    for id in num_list:
        yield id

color=next(get_color)
get_color = color_gen(len(countries))
#note generator: in a loop and has yield instead of return -> gives you one at a time
'''

#Summary statistics
def summary_stats(df):
    summary = df.describe()
    #summary.drop(columns = ['Community_Area_Number'], inplace = True)
    summary = summary.transpose()
    summary = summary.round(2)
    return summary

#not very helful
#summary_stats(df)

#analysis
def ols(use_df, y, x1, x2, x3):
    print('Dependent Variable: ' + y) #for output display
    m = smf.ols(y + '~'+x1+' + '+x2+' + '+x3, data = use_df)
    result = m.fit()
    print(result.summary()) #show results in output
    print( ) #for output display readability
    #prepare results to save to png
    #plt.rc('figure',figsize=(9, 5.5))
    #plt.text(0.01, 0.05, str(result.summary()), {'fontsize': 10}, fontproperties = 'monospace')
    #plt.axis('off')
    #plt.tight_layout()
    #plt.title('Death by '+ y +' on Green-Space, Controling for Area SES and Health Centers')

#cite https://stackoverflow.com/questions/46664082/save-statsmodels-results-in-python-as-image-file
#cite https://matplotlib.org/users/tight_layout_guide.html

ols(df, 'final_finding_id', 'race_id', 'victim_race_id', 'complaint_category_id')

#Explore covariate relationships:
def covt_check(df, y_string, x_string):
    print(y_string + ' on ' + x_string)
    test_model = smf.ols(y_string + '~' + x_string , data = df)
    result = test_model.fit()
    print(result.summary())
    print( )

#this isn't valid though bc id is read as continuous but it's catagorical
covt_check(df, 'race_id', 'victim_race_id')


#ok, that's interesting, if the compliant is sustained, the officer is less likely to be white
covt_check(df, 'white_officer', 'sustained')

df['race_id'].dtype


df = dummy_maker(df, 'victim_race', 'white_victim', 'WHITE')
covt_check(df, 'white_victim', 'sustained')
#sustained, more likelyto be a white victim
covt_check(df, 'sustained', 'white_victim')


df = dummy_maker(df, 'final_discipline', 'no_action_discipline', 'NO ACTION TAKEN')
covt_check(df, 'no_action_discipline', 'white_victim')
#white victim less likely no discipline (could be bc more likely sustained)

covt_check(df, 'no_action_discipline', 'white_officer')
#white offecer more likely no discipline (could also be bc less likely sustained)

ols(df, 'no_action_discipline', 'white_officer', 'white_victim', 'sustained')



df = dummy_maker(df, 'gender', 'male_officer', 'MALE')
df = dummy_maker(df, 'victim_gender', 'male_victim', 'MALE')

check_new_col(df, 'gender', 'male_officer') #debug

#sex not stat sig, even with all this data
ols(df, 'no_action_discipline', 'white_officer', 'male_officer', 'sustained')

#stat sig, but only at 0.005, I've been seeing 0.00 for the other univariate rlxns
covt_check(df, 'no_action_discipline', 'male_officer')
#not stat sig
covt_check(df, 'no_action_discipline', 'male_victim')
#not stat sig
covt_check(df, 'no_action_discipline', 'birth_year')

#hmm this is stat sig, and that's odd bc sustained and action are so closely linkes
covt_check(df, 'sustained', 'birth_year')
#later birth year, more a little more likely to be white
covt_check(df, 'white_officer', 'birth_year') 
#brith year stat sig, but tinny coeficient 
ols(df, 'no_action_discipline', 'white_officer', 'birth_year', 'sustained')



