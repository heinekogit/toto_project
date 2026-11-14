# v4.0 Step 4：条件Sの仮説効果と10口入替境界

条件Sを固定した6開催の試合前保存確率のみを使用。実結果は計算に使わない。倍率1.1/1.5/2.0と効果状態の重み0〜1は**仮説**であり、条件Sの推定効果・成立頻度ではない。各開催でS該当試合すべてのDオッズを同時に倍率調整した状態Qと、保存final分布Pとの混合(1-w)P+wQを評価する。状態は同一開催内で共通に成立する仮定で、試合ごとに独立して発生する誤差モデルではない。

比較は毎回3^13全組から異なる上位10口を厳密に選ぶ。`first_D_weight`は、初めてD入り券が基準10口の一券を追い越す混合重みの境界。空欄は重み1でもD入り券が入らないことを意味する。基準分布Pで失う質量(`base_loss`)と、混合分布で得る質量(`mixture_gain`)を別に保存。この境界は**設定した倍率が本当に発生する場合だけ**の損益分岐で、現実の購入推奨ではない。

## 開催別境界（%）

```text
   round S_matches  S_count  odds_factor  first_D_weight entering_D_ticket replaced_baseline_ticket  baseline_prob13  baseline_D_tickets  no_crossing_at_weight_1
toto1660         4        1          1.1             NaN                                                    0.000287                   0                     True
toto1660         4        1          1.5           65.24     AHADAAAHAHHAA            AHAAAAAHAHAAH         0.000287                   0                    False
toto1660         4        1          2.0           36.34     AHADAAAHAHHAA            AHAAAAAHAHAAH         0.000287                   0                    False
toto1663      9;13        2          1.1             NaN                                                    0.000325                   0                     True
toto1663      9;13        2          1.5           55.93     HAAAHHAAHHAHD            HAAAHHAAHHHAH         0.000325                   0                    False
toto1663      9;13        2          2.0           34.37     HAAAHHAAHHAHD            HAAAHHAAHHHAH         0.000325                   0                    False
toto1664         5        1          1.1             NaN                                                    0.000297                   0                     True
toto1664         5        1          1.5           53.55     HAAHDAHHHAHHH            HAAHHAHHHHHHA         0.000297                   0                    False
toto1664         5        1          2.0           29.80     HAAHDAHHHAHHH            HAAHHAHHHHHHA         0.000297                   0                    False
toto1665       3;7        2          1.1             NaN                                                    0.000105                   0                     True
toto1665       3;7        2          1.5           38.55     HHAHHADAHHHHH            HAAAHAHAHHHHH         0.000105                   0                    False
toto1665       3;7        2          2.0           24.07     HHAHHADAHHHHH            HAAAHAHAHHHHH         0.000105                   0                    False
toto1668         3        1          1.1             NaN                                                    0.000138                   0                     True
toto1668         3        1          1.5           75.60     AADHHHAHHAHAA            AAHHHHAHHHHAH         0.000138                   0                    False
toto1668         3        1          2.0           42.28     AADHHHAHHAHAA            AAHHHHAHHHHAH         0.000138                   0                    False
toto1670         3        1          1.1           78.16     HHDHAHHHHAAAA            HAHHAHHHHAAAA         0.000060                   0                    False
toto1670         3        1          1.5           17.37     HHDHAHHHHAAAA            HAHHAHHHHAAAA         0.000060                   0                    False
toto1670         3        1          2.0            9.77     HHDHAHHHHAAAA            HAHHAHHHHAAAA         0.000060                   0                    False
```

## 判断

条件Sの6/8は発見標本での探索結果。倍率と状態重みをこの結果から推定・採用しない。将来の試合前保存予測でS該当時のP(D)校正と効果の分布を確認するまで、境界を実際の購入判断へ変換できない。固定Pが正しければ基準上位10口が最適で、D入り券はP上の機会損失を伴う。この研究は、どの程度の効果・頻度なら券変更が合理的かを先に示す。
