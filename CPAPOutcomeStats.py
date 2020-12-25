import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt


def stratify_by_goals():
    """Utility Function -
    Splits the dataset into two xlsx spreadsheets - 1 containing patients who are meeting treatment gaols and another
    containing patients who are not meeting goals.
    Treatment goals are defined as:
    1. an average of 4h of nightly usage on the downloads we have (no requirement for the # of downloads or
    time that those downloads summarize)
    2. a machine measured AHI (AHI_flow) of less than 5 while on treatment"""

    sheets_df = pd.read_excel(io = '~/PycharmProjects/CPAPOutcomes/Full n977 (minus UARS) w dAHI use and outcomes.xlsm',
                           sheet_name=None)
    population_df = pd.read_excel(io = '~/PycharmProjects/CPAPOutcomes/Full n977 (minus UARS) w dAHI use and outcomes.xlsm',
                           sheet_name='Population')
    meets_goals_df = population_df.loc[(population_df['MACHINE_AHI_AFTER'] < 5) & (population_df['AHI_AFTER_AVERAGE_HRS'] > 4)]
    not_at_goal_df = population_df.loc[
        (population_df['MACHINE_AHI_AFTER'] >= 5) | (population_df['AHI_AFTER_AVERAGE_HRS'] < 4)]
    pat_ids_goal = meets_goals_df['PAT_ID']  # Contains all the patient IDs that are meeting treatment goals
    not_at_goal = not_at_goal_df['PAT_ID']  # Contains all the patient IDs that are NOT meeting treatment goals

    # Create excel spreadsheet that just contains the PAT_ID of patients who are, and are not, at goal
    with pd.ExcelWriter('PAT_IDs of goal or not.xlsx') as writer:
        pat_ids_goal.to_excel(writer, sheet_name="At goal")
        not_at_goal.to_excel(writer, sheet_name="Not at goal")
        writer.save()

    # Create Excel Spreadsheet with patients who are at goal
    with pd.ExcelWriter('Full (minus UARS) meeting goals.xlsx') as writer:
        for sheetName in sheets_df:
            # Includes all rows where the PAT_ID is in the listed of PAT_ID's who meet goal machine AHI and usage
            temp_df = sheets_df[sheetName].loc[sheets_df[sheetName]["PAT_ID"].isin(pat_ids_goal)]

            temp_df.to_excel(writer, sheet_name=sheetName)
            print("Meets goals: Completed " + sheetName)
        writer.save()

    # Create Excel Spreadsheet with patients who are not at goal
    with pd.ExcelWriter('Full (minus UARS) not meeting goals.xlsx') as writer:
        for sheetName in sheets_df:
            # Includes all rows where the PAT_ID is in the listed of PAT_ID's who DON'T meet goal machine AHI and usage
            temp_df = sheets_df[sheetName].loc[~sheets_df[sheetName]["PAT_ID"].isin(pat_ids_goal)]

            temp_df.to_excel(writer, sheet_name=sheetName)
            print("Not meeting goals: Completed " + sheetName)
        writer.save()


def before_after_diff(row):
    """takes a patient row and calculates the difference between pre-CPAP values and post-CPAP values

    commented out lines of code would change it's functioning to include data from outside the 9-15 month window
    e.g. 3-9 months or 15-21 months. As it is right now, returns NaN if no data in the 9-15 month window"""

    if pd.notna(row['12 Months After (Mean)']):
        return row['12 Months After (Mean)'] - row['Before Index (Mean)']
    elif pd.notna(row['6 Months After (Mean)']):
        return np.nan  # row['6 Months After (Mean)'] - row['Before Index (Mean)']  # would include data outside 9-15 mo
    elif pd.notna(row['18 Months After (Mean)']):
        return np.nan  # row['18 Months After (Mean)'] - row['Before Index (Mean)']  # same as above.
    else:
        return np.nan


def get_difference(sheets_df, sheet_label):
    """takes the database and sheet label and calculates the before CPAP to after CPAP difference as specified by the
     before_after_diff function"""
    return sheets_df[sheet_label].apply(before_after_diff, axis=1).dropna()


def get_baseline_row(row, data_complete=True):
    """takes a patient row and returns the pre-CPAP values
    data_complete flag => when true, requires that there is data for both the baseline AND post on a given row to return a
    value"""

    if data_complete:
        if pd.notna(row['12 Months After (Mean)']) and pd.notna(row['Before Index (Mean)']):
            return row['Before Index (Mean)']
        else:
            # either no baseline value or post data available
            return np.nan
    else:
        if pd.notna(row['Before Index (Mean)']):
            return row['Before Index (Mean)']
        else:
            # no baseline value available
            return np.nan


def get_baseline(sheets_df, sheet_label, data_complete=True):
    """takes the database and sheet label and gets the before CPAP results as specified by the get_baseline_row function"""
    return sheets_df[sheet_label].apply(get_baseline_row, args=(data_complete,), axis=1).dropna()


def get_post(sheets_df, sheet_label, data_complete=True):
    """takes the database and sheet label and gets the after CPAP results as specified in get_post_row function
    data_complete flag => when true, requires that there is data for both the baseline AND post on a given row to return a
    value"""
    return sheets_df[sheet_label].apply(get_post_row, args=(data_complete,), axis=1).dropna()


def get_post_row(row, data_complete=True):
    """takes a patient row and returns the post-CPAP values. Currently only returns values in the 9-15 month time period
    this could be changed by editing the other elif clauses
    data_complete flag => when true, requires that there is data for both the baseline AND post on a given row to return a
    value"""
    if data_complete:
        if pd.notna(row['Before Index (Mean)']):
            if pd.notna(row['12 Months After (Mean)']):
                return row['12 Months After (Mean)']
            elif pd.notna(row['6 Months After (Mean)']):
                return np.nan  # row['6 Months After (Mean)']  # would include data outside 9-15 mo
            elif pd.notna(row['18 Months After (Mean)']):
                return np.nan  # row['18 Months After (Mean)']  # same as above.
            else:
                return np.nan
        else:
            return np.nan  # no pre data
    else:
        if pd.notna(row['12 Months After (Mean)']):
            return row['12 Months After (Mean)']
        elif pd.notna(row['6 Months After (Mean)']):
            return np.nan  # row['6 Months After (Mean)']  # would include data outside 9-15 mo
        elif pd.notna(row['18 Months After (Mean)']):
            return np.nan  # row['18 Months After (Mean)']  # same as above.
        else:
            return np.nan


def clasify_osa_severity(row):
    """given a row = patient, return what the classification of severity of the OSA would be"""
    if pd.isna(row['DIAGNOSTIC_AHI_BEFORE']):
        return None
    elif row['DIAGNOSTIC_AHI_BEFORE'] < 15: # implicitly 5-15 because <5 is excluded from the dataset
        return 'mild'
    elif row['DIAGNOSTIC_AHI_BEFORE'] < 30:
        return 'moderate'
    else:
        return 'severe'  # implicitly 30+


def number_osa_severity(sheets_df, normalize=False):
    """returns a series with the number of severe, moderate, and mild OSA diagnoses in the series
    Normalize=true gives proportion"""
    return sheets_df['Population'].apply(clasify_osa_severity, axis=1).value_counts(normalize)


def bootstrap_CI(sample, conf=0.95):
    '''TODO: finish this - make it work, but also decide if this is the right algorithm to use
    default alpha 0.95'''
    values = sample.values
    # configure bootstrap
    n_iterations = 1000
    n_size = int(len(sample) * 0.50)  # ?
    print(n_size)
    # run bootstrap
    stats = list()
    for i in range(n_iterations):
        # prepare train and test sets
        train = resample(values, n_samples=n_size)
        test = np.array([x for x in values if x.tolist() not in train.tolist()])
        # fit model
        model = DecisionTreeClassifier()
        model.fit(train[:, :-1], train[:, -1])
        # evaluate model
        predictions = model.predict(test[:, :-1])
        score = accuracy_score(test[:, -1], predictions)
        print(score)
        stats.append(score)

    # confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))
    #return CI


def output_final_comparisons(baseline_combined, baseline_at_goal, baseline_not_goal, post_combined, post_at_goal,
                             post_not_goal, combined_diff, at_goal_difference, not_goal_difference):
    print("\nBASELINE DEMOGRAPHICS: ")
    print("Whole dataset:")
    print(baseline_combined.describe())
    print("At Goal:")
    print(baseline_at_goal.describe())
    print("Not at goal:")
    print(baseline_not_goal.describe())
    print("Comparison (goal vs not at goal): ")
    print(stats.mannwhitneyu(baseline_at_goal, baseline_not_goal))
    # Mann Whitney because the baseline data is non-normal

    print("\nPOST RESULTS")
    print("Whole dataset:")
    print(post_combined.describe())
    print("At Goal:")
    print(post_at_goal.describe())
    print("Not at goal:")
    print(post_not_goal.describe())
    print("Comparison(goal vs not at goal): ")
    print(stats.mannwhitneyu(post_at_goal, post_not_goal))
    # Mann Whitney because the baseline data is non-normal

    print("\nCOMPARISONS PRE-POST CPAP")
    print("Before/After, Whole dataset")
    print(combined_diff.describe())
    print(stats.ttest_1samp(combined_diff, 0.0))
    combined_diff_conf_int = stats.norm.interval(0.95, loc=np.mean(combined_diff),
                                                 scale=np.std(combined_diff)/np.sqrt(len(combined_diff)))
    print("95 percent CI: {lower}, {upper}\n".format(lower=round(combined_diff_conf_int[0], 2),
                                                   upper=round(combined_diff_conf_int[1], 2)))
    # print(stats.wilcoxon(combined_diff))

    print("Before/After, At goal:")
    print(at_goal_difference.describe())
    print(stats.ttest_1samp(at_goal_difference, 0.0))
    at_goal_difference_conf_int = stats.norm.interval(0.95, loc=np.mean(at_goal_difference),
                                                 scale=np.std(at_goal_difference)/np.sqrt(len(at_goal_difference)))
    print("95 percent CI: {lower}, {upper}\n".format(lower=round(at_goal_difference_conf_int[0], 2),
                                                   upper=round(at_goal_difference_conf_int[1], 2)))
    # print(stats.wilcoxon(at_goal_difference))

    print("Before/After, Not at goal:")
    print(not_goal_difference.describe())
    print(stats.ttest_1samp(not_goal_difference, 0.0))
    not_goal_difference_conf_int = stats.norm.interval(0.95, loc=np.mean(not_goal_difference),
                                                 scale=np.std(not_goal_difference)/np.sqrt(len(not_goal_difference)))
    print("95 percent CI: {lower}, {upper}\n".format(lower=round(not_goal_difference_conf_int[0], 2),
                                                   upper=round(not_goal_difference_conf_int[1], 2)))
    # print(stats.wilcoxon(not_goal_difference))

    print("Comparison(goal vs not at goal): ")
    # TTests for these as it turns out that the differences are normal.
    print(stats.ttest_ind(at_goal_difference, not_goal_difference))
    ttest_ind_CI(at_goal_difference, not_goal_difference)
    # print(stats.mannwhitneyu(at_goal_difference, not_goal_difference))



def ttest_ind_CI(dist1, dist2):
    N1 = len(dist1)
    N2 = len(dist2)
    df = (N1 + N2 - 2)
    std1 = dist1.std()
    std2 = dist2.std()
    std_N1N2 = sqrt( ((N1 - 1)*(std1)**2 + (N2 - 1)*(std2)**2) / df)
    diff_mean = dist1.mean() - dist2.mean()
    MoE = stats.t.ppf(0.975, df) * std_N1N2 * sqrt(1/N1 + 1/N2)
    print('\nThe difference between groups is {:3.2f} [{:3.2f} to {:3.2f}] (mean [95% CI])'.format(
        diff_mean, diff_mean - MoE, diff_mean + MoE))


def main():
    combined_db = '~/PycharmProjects/CPAPOutcomes/Full n977 (minus UARS) w dAHI use and outcomes.xlsm'
    at_goal_db = '~/PycharmProjects/CPAPOutcomes/Full (minus UARS) meeting goals.xlsx'
    not_goal_db = '~/PycharmProjects/CPAPOutcomes/Full (minus UARS) not meeting goals.xlsx'

    combined_df = pd.read_excel(io=combined_db, sheet_name=None)
    at_goal_df = pd.read_excel(io=at_goal_db, sheet_name=None)
    not_goal_df = pd.read_excel(io=not_goal_db, sheet_name=None)

    # stratify meeting targets vs not by severity of OSA

    print("Combined dataset: Severity of OSA")
    print(number_osa_severity(combined_df))
    print(number_osa_severity(combined_df, normalize=True))
    print("Meeting Targets dataset: Severity of OSA")
    print(number_osa_severity(at_goal_df))
    print(number_osa_severity(at_goal_df, normalize=True))
    print("Not Meeting Targets dataset: Severity of OSA")
    print(number_osa_severity(not_goal_df))
    print(number_osa_severity(not_goal_df, normalize=True))

    # Generate dataframes of each comparison

    sys_baseline_combined = get_baseline(combined_df, 'Systolic')
    sys_baseline_at_goal = get_baseline(at_goal_df, 'Systolic')
    sys_baseline_not_goal = get_baseline(not_goal_df, 'Systolic')
    sys_post_combined = get_post(combined_df, 'Systolic')
    sys_post_at_goal = get_post(at_goal_df, 'Systolic')
    sys_post_not_goal = get_post(not_goal_df, 'Systolic')
    sys_combined_diff = get_difference(combined_df, 'Systolic')
    sys_at_goal_difference = get_difference(at_goal_df, 'Systolic')
    sys_not_goal_difference = get_difference(not_goal_df, 'Systolic')

    dias_baseline_combined = get_baseline(combined_df, 'Diastolic')
    dias_baseline_at_goal = get_baseline(at_goal_df, 'Diastolic')
    dias_baseline_not_goal = get_baseline(not_goal_df, 'Diastolic')
    dias_post_combined = get_post(combined_df, 'Diastolic')
    dias_post_at_goal = get_post(at_goal_df, 'Diastolic')
    dias_post_not_goal = get_post(not_goal_df, 'Diastolic')
    dias_combined_diff = get_difference(combined_df, 'Diastolic')
    dias_at_goal_difference = get_difference(at_goal_df, 'Diastolic')
    dias_not_goal_difference = get_difference(not_goal_df, 'Diastolic')

    spo2_baseline_combined = get_baseline(combined_df, 'SP02')
    spo2_baseline_at_goal = get_baseline(at_goal_df, 'SP02')
    spo2_baseline_not_goal = get_baseline(not_goal_df, 'SP02')
    spo2_post_combined = get_post(combined_df, 'SP02')
    spo2_post_at_goal = get_post(at_goal_df, 'SP02')
    spo2_post_not_goal = get_post(not_goal_df, 'SP02')
    spo2_combined_diff = get_difference(combined_df, 'SP02')
    spo2_at_goal_difference = get_difference(at_goal_df, 'SP02')
    spo2_not_goal_difference = get_difference(not_goal_df, 'SP02')

    bmi_baseline_combined = get_baseline(combined_df, 'BMI')
    bmi_baseline_at_goal = get_baseline(at_goal_df, 'BMI')
    bmi_baseline_not_goal = get_baseline(not_goal_df, 'BMI')
    bmi_post_combined = get_post(combined_df, 'BMI')
    bmi_post_at_goal = get_post(at_goal_df, 'BMI')
    bmi_post_not_goal = get_post(not_goal_df, 'BMI')
    bmi_combined_diff = get_difference(combined_df, 'BMI')
    bmi_at_goal_difference = get_difference(at_goal_df, 'BMI')
    bmi_not_goal_difference = get_difference(not_goal_df, 'BMI')

    cr_baseline_combined = get_baseline(combined_df, 'CREAT')
    cr_baseline_at_goal = get_baseline(at_goal_df, 'CREAT')
    cr_baseline_not_goal = get_baseline(not_goal_df, 'CREAT')
    cr_post_combined = get_post(combined_df, 'CREAT')
    cr_post_at_goal = get_post(at_goal_df, 'CREAT')
    cr_post_not_goal = get_post(not_goal_df, 'CREAT')
    cr_combined_diff = get_difference(combined_df, 'CREAT')
    cr_at_goal_difference = get_difference(at_goal_df, 'CREAT')
    cr_not_goal_difference = get_difference(not_goal_df, 'CREAT')

    a1c_baseline_combined = get_baseline(combined_df, 'HBA1C')
    a1c_baseline_at_goal = get_baseline(at_goal_df, 'HBA1C')
    a1c_baseline_not_goal = get_baseline(not_goal_df, 'HBA1C')
    a1c_post_combined = get_post(combined_df, 'HBA1C')
    a1c_post_at_goal = get_post(at_goal_df, 'HBA1C')
    a1c_post_not_goal = get_post(not_goal_df, 'HBA1C')
    a1c_combined_diff = get_difference(combined_df, 'HBA1C')
    a1c_at_goal_difference = get_difference(at_goal_df, 'HBA1C')
    a1c_not_goal_difference = get_difference(not_goal_df, 'HBA1C')

    # visualization to explore distribution of baseline data
    # sns.distplot(a1c_baseline_combined)
    # plt.savefig("data dist.png")
    # plt.close()

    # Visualize distributions, plots total, at goal, and not at goal all on the same axis for each metric
    # TODO: figure out why the colors won't change here.

    f, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=False)

    for data in [sys_combined_diff, sys_at_goal_difference, sys_not_goal_difference]:
        sns.distplot(data, ax=axes[0,0])
    axes[0, 0].set(title="pre-post systolic difference", xlabel="change, mmHg", ylabel="relative frequency")
    for data in [dias_combined_diff, dias_at_goal_difference, dias_not_goal_difference]:
        sns.distplot(data, ax=axes[0,1])
    axes[0, 1].set(title="pre-post diastolic difference", xlabel="change, mmHg", ylabel="relative frequency")
    for data in [spo2_combined_diff, spo2_at_goal_difference, spo2_not_goal_difference]:
        sns.distplot(data, ax=axes[0,2])
    axes[0, 2].set(title="pre-post spo2 difference", xlabel="change, percent", ylabel="relative frequency")
    for data in [bmi_combined_diff, bmi_at_goal_difference, bmi_not_goal_difference]:
        sns.distplot(data, ax=axes[1,0])
    axes[1, 0].set(title="bmi pre-post difference", xlabel="change, m2/kg", ylabel="relative frequency")
    for data in [cr_combined_diff, cr_at_goal_difference, cr_not_goal_difference]:
        sns.distplot(data, ax=axes[1,1])
    axes[1, 1].set(title="creat pre-post difference", xlabel="change, mg/dl", ylabel="relative frequency")
    for data in [a1c_combined_diff, a1c_at_goal_difference, a1c_not_goal_difference]:
        sns.distplot(data, ax=axes[1,2])
    axes[1, 2].set(title="a1c pre-post difference", xlabel="change, percent", ylabel="relative frequency")
    f.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("combined difference histos.png")
    plt.close()

    # output

    print("\nSYSTOLIC BP\n")
    output_final_comparisons(sys_baseline_combined, sys_baseline_at_goal, sys_baseline_not_goal, sys_post_combined,
                             sys_post_at_goal, sys_post_not_goal, sys_combined_diff, sys_at_goal_difference,
                             sys_not_goal_difference)

    print("\nDiastolic BP\n")
    output_final_comparisons(dias_baseline_combined, dias_baseline_at_goal, dias_baseline_not_goal, dias_post_combined,
                             dias_post_at_goal, dias_post_not_goal, dias_combined_diff, dias_at_goal_difference,
                             dias_not_goal_difference)

    print("\nSpO2\n")
    output_final_comparisons(spo2_baseline_combined, spo2_baseline_at_goal, spo2_baseline_not_goal, spo2_post_combined,
                             spo2_post_at_goal, spo2_post_not_goal, spo2_combined_diff, spo2_at_goal_difference,
                             spo2_not_goal_difference)

    print("\nBMI\n")
    output_final_comparisons(bmi_baseline_combined, bmi_baseline_at_goal, bmi_baseline_not_goal, bmi_post_combined,
                             bmi_post_at_goal, bmi_post_not_goal, bmi_combined_diff, bmi_at_goal_difference,
                             bmi_not_goal_difference)

    print("\nCreatinine\n")
    output_final_comparisons(cr_baseline_combined, cr_baseline_at_goal, cr_baseline_not_goal, cr_post_combined,
                             cr_post_at_goal, cr_post_not_goal, cr_combined_diff, cr_at_goal_difference,
                             cr_not_goal_difference)

    print("\nA1c\n")
    output_final_comparisons(a1c_baseline_combined, a1c_baseline_at_goal, a1c_baseline_not_goal, a1c_post_combined,
                             a1c_post_at_goal, a1c_post_not_goal, a1c_combined_diff, a1c_at_goal_difference,
                             a1c_not_goal_difference)

if __name__ == '__main__':
    main()