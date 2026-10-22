import argparse,asyncio,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.workbench_smoke import run
from asr_server.workbench.store import WorkbenchStore
from httpx import AsyncClient

def main():
 original=WorkbenchStore.create_user
 def create_with_isolated_alias(self,*args,**kwargs):
  result=original(self,*args,**kwargs)
  glossary=self.glossary()
  for entry in glossary["entries"]:
   if entry["canonical"]=="Qwen3-ASR":
    entry["aliases"]=["千问 ASR"]
    entry["auto_normalize"]=True
  glossary["version"]+=1
  with self.connection() as db:
   db.execute("UPDATE metadata SET payload=? WHERE key=?",(json.dumps(glossary,ensure_ascii=False),"glossary"))
  return result
 WorkbenchStore.create_user=create_with_isolated_alias
 observed={}
 original_get=AsyncClient.get
 async def capture(self,*args,**kwargs):
  response=await original_get(self,*args,**kwargs)
  if response.status_code==200 and "/api/jobs/" in str(response.url) and "/export" not in str(response.url) and "/audio" not in str(response.url):
   data=response.json()
   result=data.get("result") or {}
   if result.get("timestamps") and result.get("version")==1:
    observed["timestamp_sources"]=sorted({item["source"] for item in result["timestamps"]})
    observed["timestamp_count"]=len(result["timestamps"])
  return response
 AsyncClient.get=capture
 args=argparse.Namespace(mock=False,model="qwen3-asr-1.7b",audio=Path("test-fixtures/audio/test_short.wav"),timeout=600)
 report=asyncio.run(run(args))
 assert report["text"]!=report["raw_text"]
 assert "Qwen3-ASR" in report["text"] and "千问 ASR" in report["raw_text"]
 assert report["quality"]["changes"]
 assert observed["timestamp_sources"]==["qwen3_forced_aligner"]
 report["isolated_alias"]={"canonical":"Qwen3-ASR","aliases":["千问 ASR"],"auto_normalize":True,"scope":"temporary smoke database only"}
 report["observed_alignment"]=observed
 Path("data/validation/workbench-17b-alias-smoke.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":
 main()
