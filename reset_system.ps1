# A-MEM System Reset Script
# Deletes all stored data (ChromaDB, Graph, Lock-Files)
#
# ⚠️  IMPORTANT: After reset, the MCP Server must be restarted!
# The graph is loaded on server start and remains in memory.
# Only a server restart ensures a truly empty graph.

Write-Host "🔄 Resetting A-MEM system completely..." -ForegroundColor Yellow
Write-Host ""

# Delete ChromaDB
if (Test-Path "data\chroma") {
    Remove-Item -Recurse -Force "data\chroma"
    Write-Host "  ✅ ChromaDB deleted" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  ChromaDB does not exist" -ForegroundColor Gray
}

# Delete Graph
if (Test-Path "data\graph\knowledge_graph.json") {
    Remove-Item -Force "data\graph\knowledge_graph.json"
    Write-Host "  ✅ Graph deleted" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Graph does not exist" -ForegroundColor Gray
}

# Delete Lock-File
if (Test-Path "data\graph\graph.lock") {
    Remove-Item -Force "data\graph\graph.lock"
    Write-Host "  ✅ Lock-File deleted" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Lock-File does not exist" -ForegroundColor Gray
}

Write-Host ""
Write-Host "✅ Files deleted" -ForegroundColor Green
Write-Host ""
Write-Host "⚠️  IMPORTANT NOTE:" -ForegroundColor Yellow
Write-Host "   The MCP Server must be RESTARTED!" -ForegroundColor Yellow
Write-Host "   The graph is loaded on server start and remains in memory." -ForegroundColor Yellow
Write-Host "   Only a server restart ensures a truly empty graph." -ForegroundColor Yellow
Write-Host ""
Write-Host "   In Cursor: Reload MCP Server (Cursor Settings → MCP → Restart)" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Verification:" -ForegroundColor Cyan

if (Test-Path "data\chroma") {
    Write-Host "  ❌ ChromaDB still exists" -ForegroundColor Red
} else {
    Write-Host "  ✅ ChromaDB deleted" -ForegroundColor Green
}

if (Test-Path "data\graph\knowledge_graph.json") {
    Write-Host "  ❌ Graph still exists" -ForegroundColor Red
} else {
    Write-Host "  ✅ Graph deleted" -ForegroundColor Green
}

if (Test-Path "data\graph\graph.lock") {
    Write-Host "  ❌ Lock-File still exists" -ForegroundColor Red
} else {
    Write-Host "  ✅ Lock-File deleted" -ForegroundColor Green
}
