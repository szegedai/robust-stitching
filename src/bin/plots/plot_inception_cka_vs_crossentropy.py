import pandas as pd
from matplotlib import pyplot as plt
import os
import sys
import seaborn as sns
import json
import argparse
from dotmap import DotMap
import numpy as np

# ================================================================
# SETTINGS
# ================================================================

settings = DotMap()

# PLOT
# ----------------------------------------
settings.plot.colors = {
    'frank_crossentropy'  : '#F00',
    'frank_cka'  : '#F60',
    'ps_inv_crossentropy' : '#00F',
    'ps_inv_cka' : '#06F'
}
settings.plot.rename_dict = {
    'frank_crossentropy' : 'With Task Loss - Cross-entropy',
    'ps_inv_crossentropy' : 'Linear Least Squares - Cross-entropy',
    'frank_cka' : 'With Task Loss - CKA',
    'ps_inv_cka' : 'Linear Least Squares - CKA'
}
settings.plot.linewidth = 5
settings.plot.ce_linestyle = '-'
settings.plot.cka_linestyle = '--'
settings.plot.x_label = 'Layer'
settings.plot.y_label_left = 'CKA between stiched layers'
settings.plot.y_label_right = 'Cross-entropy'

# MEAN & STD TABLE
# ----------------------------------------
settings.mean_std_table.caption = 'Mean metrics with standard deviations in parentheses.'
settings.mean_std_table.column_names = ['Cross-entropy', 'CKA'] # Order cannot be changed this way
settings.mean_std_table.row_names = ['Ps. Inv', 'Frank'] # Order cannot be changed this way

# LAYERWISE TABLE
settings.layerwise_table.caption = 'Effect on logits, layerwise split on Frank stitches.'
settings.layerwise_table.column_names = {'crossentropy' : 'Cross-entropy', 'cka' : 'CKA'}

# Layers to process:
layers_to_process = [
    # 'conv2d0_pre_relu',
    # 'conv2d1_pre_relu',
    # 'conv2d2_pre_relu',
    'mixed3a_pre_relu',
    'mixed3b_pre_relu',
    'mixed4a_pre_relu',
    'mixed4b_pre_relu',
    'mixed4c_pre_relu',
    'mixed4d_pre_relu',
    'mixed4e_pre_relu',
    'mixed5a_pre_relu',
    'mixed5b_pre_relu',
]

# ================================================================
# PROCESSING
# ================================================================


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('--csv', default="results/find_trans_inception/summary.csv")
    parser.add_argument('--out-dir', default='results/find_trans_inception/plots/cka_ce/')
    return parser.parse_args(args)

def filter_df(df):
    filters = [
        df.l1 == 0.,
        df.target_type=='soft_2',
        df.init =='ps_inv',
        df.front_layer.isin(layers_to_process),
        df.end_layer.isin(layers_to_process),
        df.front_model != df.end_model,
        df.temperature == 1.0,
    ]
    return df[np.logical_and.reduce(filters)]

def get_df(csv, out_dir):

    # Filter csv to relevant parts
    filtered_df = filter_df(pd.read_csv(csv))
    filtered_df['front_layer'] = filtered_df['front_layer'].str[:-9] #.capitalize()
    filtered_df = filtered_df.sort_values(['front_layer']).copy()
    
    # Calculate accuracies
    df = filtered_df.copy()
    df['frank_cka'] = df['cka_frank']
    df['ps_inv_cka'] = df['cka_ps_inv']
    df['frank_crossentropy'] = df['after_crossentropy']
    df['ps_inv_crossentropy'] = df['ps_inv_crossentropy']

    # Rename columns in local dataframe
    # df = df.rename(columns={
    #     'after_cka' : 'frank_cka'
    # })

    # Group values into one column with a category column
    def make_one_col(column):
        new_df = df[['front_layer', column]].copy()
        new_df['matrix'] = column
        new_df['style'] = 'crossentropy' if 'crossentropy' in column else 'cka'
        new_df = new_df.rename(columns={column : 'value'})
        return new_df
    dfs = [
        make_one_col('frank_crossentropy'),
        make_one_col('frank_cka'),
        make_one_col('ps_inv_crossentropy'),
        make_one_col('ps_inv_cka'),
    ]
    sum_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    
    # Save
    filtered_df.to_csv(os.path.join(out_dir, 'filtered_df.csv'), index=False)
    sum_df.to_csv(os.path.join(out_dir, 'sum_df.csv'), index=False)
    
    return filtered_df, sum_df

def save_table_mean_std(df, out_dir):
    
    global settings
    conf = settings.mean_std_table
    out_file = os.path.join(out_dir, 'overall_mean_std.tex')
    
    # Generatre mean and std
    df = df.copy()
    df = df.groupby(['matrix'])['value'].describe()[['mean', 'std']].copy()
    
    # Create table in desired format
    _mean = lambda x: f"{df.loc[x, 'mean']:0.3f}"
    _std  = lambda x: f"(\pm{df.loc[x, 'std']:0.3f})"
    _cell = lambda x: f"{_mean(x)} {_std(x)}"
    _row  = lambda x: [_cell(x+'_crossentropy'), _cell(x+'_cka')]
    new_df = pd.DataFrame({
        conf.row_names[0] :  _row('ps_inv'),
        conf.row_names[1] :  _row('frank')
    }, index=conf.column_names)
    
    # Convert table to latex
    table_latex = new_df.to_latex(escape=False, column_format='l c c')
    
    # Place the latex table in a figure and add captopn
    latex = "\\begin{table}\n\\centering\n" + table_latex + \
            "   \\caption{" + conf.caption + "}\n" + \
            "   \\label{fig:my_label}\n" + \
            "\\end{table}"
    
    # Save
    with open(out_file, "w") as text_file:
        print(latex, file=text_file)


def save_table_layerwise(df, out_dir):

    global settings
    conf = settings.layerwise_table
    out_file = os.path.join(out_dir, 'layerwise_mean_std.tex')

    df = df.copy()
    
    # Create CKA/Acc and PsInv/Frank categories
    df['mode'] = df.matrix.apply(lambda x: x.split('_')[-1])
    df['method'] = df.matrix.apply(lambda x: '_'.join(x.split('_')[:-1]))

    # Available layers in order
    layers = df['front_layer'].drop_duplicates().sort_values()
    
    # Create dataframe
    new_df = pd.DataFrame(index=layers)
    for layer in layers:
        for mode in df['mode'].drop_duplicates():

            # Filter ps_inv and frank
            subdf = df[(df.front_layer==layer)&(df['mode']==mode)]
            ps_inv = subdf[subdf.method == 'ps_inv']['value'].reset_index(drop=True)
            frank = subdf[subdf.method == 'frank']['value'].reset_index(drop=True)

            # Get mode spcific changes (e.g. % mark)
            mark = '\%' if mode=='cka' else ''
            multiplier = 100 if mode=='cka' else 1

            # Caulculate mean and std
            mean = (frank-ps_inv).mean() * multiplier
            std = (frank-ps_inv).std() * multiplier

            # Insert variable in table
            val = f"{mean:1.3f}{mark} (\pm{std:1.3f}{mark})"
            new_df.loc[layer, mode] = val

    # Final decoration on table  
    new_df.index.name = None
    new_df = new_df.rename(columns=conf.column_names)
    
    # Convert table to latex
    table_latex = new_df.to_latex(escape=False, column_format='l c c')

    # Place the latex table in a figure and add captopn
    latex = "\\begin{table}\n\\centering\n" + table_latex + \
            "   \\caption{" + conf.caption + "}\n" + \
            "   \\label{fig:my_label}\n" + \
            "\\end{table}"

    # Save
    with open(out_file, "w") as text_file:
        print(latex, file=text_file)

def save_diagram(df, out_dir):

    global settings
    conf = settings.plot
    out_file = os.path.join(out_dir,'crossentropy_vs_cka.pdf')

    fig = plt.figure(figsize=(16,9));

    subdf = df[df['style']=='cka']
    cka_ax = sns.lineplot(
        data=subdf, #kind="line",
        x="front_layer", y="value",
        hue="matrix", style='style',
        palette=conf.colors,
        linewidth=conf.linewidth);

    ce_ax = plt.twinx()
    subdf = df[df['style']!='cka']
    ce_ax = sns.lineplot(
        data=subdf, #kind="line",
        x="front_layer", y="value",
        hue="matrix", style='style',
        palette=conf.colors,
        linewidth=conf.linewidth, ax=ce_ax);


    xlabels = df['front_layer'].drop_duplicates().str.replace('.add', '').tolist()
    xlabels.sort()

    def set_linestyle(ax, linesetyle):

        # Remove columns from legend
        h,l = ax.get_legend_handles_labels()
        cols_to_remove = ['matrix', 'style', 'crossentropy', 'cka']
        h = [x for (x,y) in zip(h,l) if y not in cols_to_remove]
        l = [x for x in l if x not in cols_to_remove]

        # Set linewidth in legend
        for x in h:
            x.set_linewidth(conf.linewidth)

        # Set linestyles of CKA
        h[0].set_linestyle(linesetyle)
        h[1].set_linestyle(linesetyle)

        ax.lines[0].set_linestyle(linesetyle)
        ax.lines[1].set_linestyle(linesetyle)

        return h, l

    h1, l1 = set_linestyle(cka_ax, conf.cka_linestyle)
    h2, l2 = set_linestyle(ce_ax, conf.ce_linestyle)
    h, l = h1+h2, l1+l2

    cka_ax.get_legend().remove()
    ce_ax.get_legend().remove()

    # Remove sns default legend and set custom
    # g._legend.remove()
    legends = plt.legend(h,l,loc='center', fontsize=20)
    for i in range(4):
        legend = legends.get_texts()[i]
        title = legend.get_text()
        new_title = conf.rename_dict[title]
        legend.set_text(new_title)

    
    cka_ax.set_ylabel(conf.y_label_left, size = 24)
    cka_ax.set_xlabel(conf.x_label, size = 24)
    cka_ax.set_xticklabels(xlabels, size=24, rotation=-45)
    cka_ax.tick_params(axis='y', labelsize=24)
    cka_ax.set_ylim([0.6 ,1])

    cka_ax.xaxis.grid()

    # Set grids
    # plt.grid()

    ce_ax.set_ylabel(conf.y_label_right, size = 24)
    ce_ax.set_xlabel(conf.x_label, size = 24)
    ce_ax.set_ylim([0,0.5])

    ce_ax.xaxis.grid()

    plt.tick_params(axis='x', labelsize=24, rotation=-45)
    plt.tick_params(axis='y', labelsize=24, labelright=True)

    # Save
    plt.savefig(out_file, bbox_inches='tight')

    #plt.show()
    

def save_report(filtered_df, out_dir):
    data = DotMap()
    data.no_experiments = len(filtered_df)
    data.unique_networks = list(set(filtered_df.front_model).union(set(filtered_df.end_model)))
    data.same_networks_compared = str((filtered_df.front_model == filtered_df.end_model).any())
    
    extra_cols = ['l1', 'target_type', 'weight_decay', 'init', 'temperature']
    for col in extra_cols:
        data[col] = filtered_df[col].drop_duplicates().tolist()
    data.layers = filtered_df.front_layer.drop_duplicates().tolist()
    data.layers.sort()
    
    with open(os.path.join(out_dir, 'run_settings.json'), 'w') as fp:
        json.dump(data, fp)
        
    return data


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    conf = parse_args(args)

    # Create directory if not exist
    os.makedirs(conf.out_dir, exist_ok=True)

    # Read in original csv
    filtered_df, df = get_df(conf.csv, conf.out_dir)

    # Run and save measurements
    save_table_layerwise(df, conf.out_dir)
    save_report(filtered_df, conf.out_dir)
    save_table_mean_std(df, conf.out_dir)
    save_diagram(df, conf.out_dir)

    print(f'Results saved at {conf.out_dir}')
    

if __name__ == '__main__':
    main()