@echo off

:: 创建目录结构
mkdir project_root
cd project_root
mkdir data models utils experiments results

:: 生成除csv文件外的所有py空文件
cd models
echo. > __init__.py
echo. > gcn_model.py
echo. > gat_model.py
echo. > graph_conv_model.py
cd ..

cd utils
echo. > __init__.py
echo. > data_loader.py
cd ..

cd experiments
echo. > train_gcn.py
echo. > train_gat.py
echo. > train_graph_conv.py
cd ..

cd results
mkdir gcn
mkdir gat
mkdir graph_conv
cd ..

echo. > main.py

echo 生成目录结构和空文件完成
pause
