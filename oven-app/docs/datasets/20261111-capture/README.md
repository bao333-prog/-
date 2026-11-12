# 西兰花、薯条和玉米新增实拍记录

采集人：@MascoTxTs。配套记录中的采集日期为2026-11-11，整理日期为2026-11-12。

我补充了西兰花、薯条和玉米的实拍数据，按食材形态、数量、有无锡纸和上中下层安排采集，计划每种状态拍摄30张。

## 实际工作量

| 食材类别 | 本批新增JPG |
|---|---:|
| 西兰花 | 302 |
| 薯条 | 494 |
| 玉米（含普通玉米、糯玉米） | 536 |
| 合计 | 1332 |

新增JPG均为1280×720，并有对应JSON。文件夹另有249张旧PNG，不计入本次新增。目录和ZIP是同批文件的两种保存形式，不重复计算。本次按新增JPG及采集日期识别批次，未将名称相近的旧PNG合并进新批次。

“玉米”作为总类，普通整根、截块和整根糯玉米作为形态区分；单个/两个西兰花、单根/两根/一堆薯条也是采集形态，不等同于新增模型类别。

## 采集覆盖核对

按上述8种具体形态 × 2种锡纸状态 × 3层，每状态30张，计划是48种状态、1440张。这只是计划口径，不能根据总数差额直接认定缺拍108张：部分文件名可能混用了形态或层位。

本批有38个实际命名组，其中20组恰好30张，18组数量不同。完整计划对照见[coverage.csv](coverage.csv)，逐组实际数量见[groups.csv](groups.csv)。

- Broccoli_two_yes有61张，文件名缺少层位，不能直接分配到上中下层，也不能凭数量判断其中是否包含单个西兰花。
- 新JPG中没有明确的单个西兰花有锡纸分组，也没有单根薯条独立分组；这表示命名清单未能确认，不等于照片一定未拍。
- chips_two分组有28、53、59、60张等情况，需核对是否混入单根薯条或多次采集。
- 整根糯玉米有锡纸high组62张，未找到明确的有锡纸low组，不能擅自把其中32张改成low。
- 一堆薯条无锡纸low组24张、普通玉米截块无锡纸high组26张，需要结合实物及拍摄记录复核。

## 记录文件

- [manifest.csv](manifest.csv)：1332张新增图片的相对文件名、类别、名称解析条件、采集日期、尺寸和SHA-256。
- [statistics.json](statistics.json)：总量、日期、尺寸、命名组和检查统计。
- [质检与使用说明](质检与使用说明.md)：检查结果、待确认项和使用方式。
- [issues.csv](issues.csv)：混合目录内旧PNG的配套问题，均不属于新增JPG。

## 实际命名组

| 命名组 | 实际张数 | 与30张差值 |
|---|---:|---:|
| Broccoli_one_no_high | 30 | +0 |
| Broccoli_one_no_low | 32 | +2 |
| Broccoli_one_no_mid | 29 | -1 |
| Broccoli_two_no_high | 29 | -1 |
| Broccoli_two_no_low | 31 | +1 |
| Broccoli_two_no_mid | 30 | +0 |
| Broccoli_two_yes | 61 | +31 |
| Broccoli_two_yes_high | 30 | +0 |
| Broccoli_two_yes_mid | 30 | +0 |
| chips_much_no_high | 30 | +0 |
| chips_much_no_low | 24 | -6 |
| chips_much_no_mid | 30 | +0 |
| chips_much_yes_high | 30 | +0 |
| chips_much_yes_low | 30 | +0 |
| chips_much_yes_mid | 30 | +0 |
| chips_two_no_high | 28 | -2 |
| chips_two_no_low | 60 | +30 |
| chips_two_no_mid | 59 | +29 |
| chips_two_yes_high | 53 | +23 |
| chips_two_yes_low | 60 | +30 |
| chips_two_yes_mid | 60 | +30 |
| corn_one_high_sep_no | 26 | -4 |
| corn_one_high_sep_yes | 30 | +0 |
| corn_one_high_whole_have | 30 | +0 |
| corn_one_high_whole_no | 29 | -1 |
| corn_one_low_sep_no | 31 | +1 |
| corn_one_low_sep_yes | 30 | +0 |
| corn_one_low_whole_have | 30 | +0 |
| corn_one_low_whole_no | 29 | -1 |
| corn_one_mid_sep_no | 30 | +0 |
| corn_one_mid_sep_yes | 30 | +0 |
| corn_one_mid_whole_have | 30 | +0 |
| corn_one_mid_whole_no | 30 | +0 |
| glutinouscorn_one_high_whole_no | 30 | +0 |
| glutinouscorn_one_high_whole_yes | 62 | +32 |
| glutinouscorn_one_low_whole_no | 29 | -1 |
| glutinouscorn_one_mid_whole_no | 30 | +0 |
| glutinouscorn_one_mid_whole_yes | 30 | +0 |
