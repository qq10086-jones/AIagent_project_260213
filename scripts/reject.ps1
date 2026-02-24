param(
  [Parameter(Mandatory=$true)][string]$TaskId,
  [string]$Token = "dev-approval-token"
)

$base = "http://localhost:3000"
Write-Host "Rejecting $TaskId ..."

$resp = curl.exe -s -X POST "$base/tasks/$TaskId/reject" -H "X-Approval-Token: $Token"
Write-Host $resp

Write-Host "Fetch task ..."
curl.exe -s "$base/tasks/$TaskId"
