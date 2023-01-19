import pandas as pd
df = pd.read_csv("result_with_alpha_JAN19.csv")
trial_lst = df['trial'].unique()
print(trial_lst)
bias_name_lst = df['name'].unique()
print(bias_name_lst)
selected_alpha_df = pd.DataFrame(columns=df.columns)
for b in bias_name_lst:
    for t in trial_lst:
        curr_df = df.loc[df['trial'] == t].loc[df['name'] == b]
        max_idx = curr_df['acc'].argmax()
        # print(curr_df.iloc[max_idx])
        selected_alpha_df = selected_alpha_df.append(curr_df.iloc[max_idx], ignore_index=True)
print(selected_alpha_df)
selected_alpha_df.to_csv('selected_alpha_JAN19.csv', index=False)
