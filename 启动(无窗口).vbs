' DeepSeekChat 无窗口启动脚本
' 双击运行,后台启动服务并自动打开浏览器

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' 获取脚本所在目录
strPath = fso.GetParentFolderName(WScript.ScriptFullName)
ChDir strPath

' 检查 Python 路径
pythonPath = "E:\Python311\python.exe"
If Not fso.FileExists(pythonPath) Then
    pythonPath = "python"
End If

' 检查模型文件 (自动扫描 models 目录下的 .gguf)
modelsDir = strPath & "\models"
hasModel = False
If fso.FolderExists(modelsDir) Then
    Set folder = fso.GetFolder(modelsDir)
    For Each file In folder.Files
        If LCase(fso.GetExtensionName(file.Name)) = "gguf" Then
            hasModel = True
            Exit For
        End If
    Next
End If
If Not hasModel Then
    ' 无窗口下载模型
    WshShell.Run pythonPath & " download_model.py", 0, True
End If

' 后台启动服务 (隐藏窗口)
WshShell.Run pythonPath & " " & strPath & "\app.py", 0, False

' 等待服务启动
WScript.Sleep 3000

' 打开浏览器
WshShell.Run "http://127.0.0.1:7860"
