# Flux LSP サーバー利用マニュアル

本マニュアルでは、実装された `fluno lsp` サーバーを VS Code や Neovim で使用するための設定方法を説明します。

## 1. サーバーのビルド

まず、最新のコードをコンパイルして実行バイナリを作成します。

```powershell
cargo build
```

これにより、`target/debug/fluno.exe` が生成されます。

## 2. VS Code での設定

VS Code で Flux 言語の LSP を使用するには、汎用 LSP クライアント拡張機能を使用します。以下のいずれかをインストールしてください。

### 推奨拡張機能: **"Any LSP"** (または **"Generic LSP"**)
- **作者**: `mshr-h`
- **Marketplace ID**: `mshr-h.any-lsp` (または `mshr-h.generic-lsp`)
- これらは非常にシンプルで、`settings.json` にコマンドを記述するだけで動作します。

### 設定手順 (`settings.json`)

1.  VS Code で `Ctrl + ,` (Windows) または `Cmd + ,` (Mac) を押し、設定を開きます。
2.  右上の「設定 (JSON) を開く」アイコンをクリックします。
3.  以下の設定を追加します。**注意：前の項目の後ろにカンマ（`,`）があることを確認してください。**

```json
{
  "any-lsp.servers": {
    "flux": {
      "command": "C:/Users/froms/Desktop/Flux-main/target/debug/fluno.exe",
      "args": ["lsp"],
      "filetypes": ["fln"],
      "rootIdentifier": [".git", "Cargo.toml"]
    }
  }
}
```

※ **"Any LSP"** を使用する場合、キーは `"any-lsp.servers"` になります。
※ もし **"Generic LSP"** を使用する場合は `"generic-lsp.servers"` に読み替えてください。

※ `command` のパスは、実際の環境に合わせて absolute path で指定してください。

## 3. Neovim (nvim-lspconfig) での設定

Neovim を使用している場合は、以下の Lua 設定を `init.lua` 等に追加することで接続可能です。

```lua
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

if not configs.flux_lsp then
  configs.flux_lsp = {
    default_config = {
      cmd = { "C:/Users/froms/Desktop/Flux-main/target/debug/fluno.exe", "lsp" },
      filetypes = { 'fln' },
      root_dir = lspconfig.util.root_pattern(".git", "Cargo.toml"),
      settings = {},
    },
  }
end

lspconfig.flux_lsp.setup{}
```

## 4. 利用可能な機能

設定が完了すると、`.fln` ファイルを開いた際に以下の機能が有効になります。

- **リアルタイム診断**: 構文エラーや型エラーが波線で表示されます。
- **ホバー**: 関数名や型の上にマウスを置くと詳細情報が表示されます。
- **定義へ移動**: `F12` キーなどでシンボルの定義場所にジャンプできます。
- **アウトライン**: エディタの「アウトライン」ビューにファイル内の構造が表示されます。
- **コード補完**: `fn` などのキーワードや定義済みのシンボルが補完候補に出ます。
