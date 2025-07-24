from pathlib import Path

import pandas as pd

df = pd.read_csv("report_output/dataset_statistics_by_column.csv")



info_cols = ["VM", "count", "start", "end"]
cpu_cols = ["VM", "CPU_USAGE_MHZ_mean", "CPU_USAGE_MHZ_median", "CPU_USAGE_MHZ_std",
            "CPU_USAGE_MHZ_min", "CPU_USAGE_MHZ_q25", "CPU_USAGE_MHZ_q75", "CPU_USAGE_MHZ_max"]
mem_cols = ["VM", "MEMORY_USAGE_KB_mean", "MEMORY_USAGE_KB_median", "MEMORY_USAGE_KB_std",
            "MEMORY_USAGE_KB_min", "MEMORY_USAGE_KB_q25", "MEMORY_USAGE_KB_q75", "MEMORY_USAGE_KB_max"]
disk_cols = ["VM", "NODE_1_DISK_IO_RATE_KBPS_mean", "NODE_1_DISK_IO_RATE_KBPS_median", "NODE_1_DISK_IO_RATE_KBPS_std",
             "NODE_1_DISK_IO_RATE_KBPS_min", "NODE_1_DISK_IO_RATE_KBPS_q25", "NODE_1_DISK_IO_RATE_KBPS_q75", "NODE_1_DISK_IO_RATE_KBPS_max"]
net_cols = ["VM", "NODE_1_NETWORK_TR_KBPS_mean", "NODE_1_NETWORK_TR_KBPS_median", "NODE_1_NETWORK_TR_KBPS_std",
            "NODE_1_NETWORK_TR_KBPS_min", "NODE_1_NETWORK_TR_KBPS_q25", "NODE_1_NETWORK_TR_KBPS_q75", "NODE_1_NETWORK_TR_KBPS_max"]


for col in cpu_cols[1:]:
    df[col] = df[col] / 1024

for col in mem_cols[1:]:
    df[col] = df[col] / (1024 ** 3)

for col in disk_cols[1:]:
    df[col] = df[col] / 1024

for col in net_cols[1:]:
    df[col] = df[col] / 1024


def df_to_latex_table(df_subset, caption, label, col_names=None):
    df_copy = df_subset.copy()

    if "VM" in df_copy.columns:
        df_copy["VM"] = df_copy["VM"].apply(lambda x: x.replace("_", "\\_"))

    numeric_cols = df_copy.select_dtypes(include="number").columns
    df_copy[numeric_cols] = df_copy[numeric_cols].round(2)

    df_latex = df_copy.to_latex(index=False,
                                column_format="|l|" + "c|" * (df_copy.shape[1] - 1),
                                float_format="%.2f",
                                caption=caption,
                                label=label,
                                longtable=False,
                                escape=False)




    lines = df_latex.splitlines()
    for i in range(6, len(lines)):
        if not lines[i].startswith("\\"):
            lines[i] = lines[i] + r" \hline"
    return "\n".join(lines)


Path("tables").mkdir(exist_ok=True)

with open("report_output/vm_info.tex", "w") as f:
    f.write(df_to_latex_table(df[info_cols], "Informacje ogólne o wybranych maszynach wirtualnych", "tab:vm-info"))

with open("report_output/vm_cpu.tex", "w") as f:
    f.write(df_to_latex_table(df[cpu_cols], "Statystyki wykorzystania CPU (w MHz)", "tab:vm-cpu"))

with open("report_output/vm_mem.tex", "w") as f:
    f.write(df_to_latex_table(df[mem_cols], "Statystyki wykorzystania pamięci (w KB)", "tab:vm-mem"))

with open("report_output/vm_disk.tex", "w") as f:
    f.write(df_to_latex_table(df[disk_cols], "Statystyki użycia dysku (KB/s)", "tab:vm-disk"))

with open("report_output/vm_net.tex", "w") as f:
    f.write(df_to_latex_table(df[net_cols], "Statystyki transferu sieciowego (KB/s)", "tab:vm-net"))
