import urllib.request
import json
import base64

# 这是你要请求的 API 地址
url = "http://localhost:18081/run"

# 这里填入你的请求载荷（替换成你最新的图片路径）
payload = {
    "op": "artifact.pull",
    "args": {
        "container": "infra-openclaw-1",
        "path": "/home/node/.openclaw/media/browser/0b856d7d-6da0-4f81-a1ff-5570bec56931.png"
    }
}

# 发送 HTTP POST 请求
req_body = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=req_body, headers={'Content-Type': 'application/json'})

print("正在拉取图片，请稍候...")
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode('utf-8'))
    
    if result.get("ok"):
        # 魔法反转：把 Base64 字符串解码回二进制图片数据
        img_data = base64.b64decode(result["base64_data"])
        filename = result["file_name"]
        
        # 写入到你本地的 Windows 文件夹里
        with open(filename, "wb") as f:
            f.write(img_data)
        print(f"✅ 大功告成！图片已成功保存到当前目录: {filename}")
    else:
        print("❌ 拉取失败:", result)