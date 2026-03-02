const path = require('path');
const fs = require('fs');
const { workspace, window, ExtensionContext } = require('vscode');
const { LanguageClient, LanguageClientOptions, ServerOptions, TransportKind } = require('vscode-languageclient/node');

let client;

function activate(context) {
    console.log('Fluno Extension Activate: Starting...');

    const outputChannel = window.createOutputChannel('Fluno Language Server');
    // 自動で開くと邪魔になることがあるので、エラー時以外は非表示でも良いかも
    // outputChannel.show(true); 
    outputChannel.appendLine('Fluno extension activation started.');

    const config = workspace.getConfiguration('fluno');
    let serverPath = config.get('executablePath');

    // 1. 設定値があればそれを使う
    if (serverPath) {
        outputChannel.appendLine(`Using configured executablePath: ${serverPath}`);
    } else {
        // 2. なければ、拡張機能に同梱されたバイナリを探す
        // __dirname は extension.js のある場所 (extensions/author.ext-ver/)
        const bundledPath = path.join(context.extensionPath, 'fluno.exe'); // Windows想定

        if (fs.existsSync(bundledPath)) {
            serverPath = bundledPath;
            outputChannel.appendLine(`Found bundled server at: ${serverPath}`);
        } else {
            // 3. 同梱もなければ開発用パス（フォールバック）
            const rootPath = workspace.workspaceFolders ? workspace.workspaceFolders[0].uri.fsPath : '';
            serverPath = path.join(rootPath, 'target', 'debug', 'fluno.exe');
            outputChannel.appendLine(`Bundled not found, trying fallback: ${serverPath}`);
        }
    }

    const serverOptions = {
        run: { command: serverPath, args: ['lsp'], transport: TransportKind.stdio },
        debug: { command: serverPath, args: ['lsp'], transport: TransportKind.stdio }
    };

    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'fluno' }],
        synchronize: {
            fileEvents: workspace.createFileSystemWatcher('**/*.fln')
        },
        outputChannel: outputChannel
    };

    try {
        client = new LanguageClient(
            'flunoLanguageServer',
            'Fluno Language Server',
            serverOptions,
            clientOptions
        );

        outputChannel.appendLine('Language Client instance created. Starting...');
        client.start().then(() => {
            outputChannel.appendLine('Language Client started successfully.');
        }).catch((e) => {
            outputChannel.appendLine(`ERROR: Start failed: ${e}`);
            window.showErrorMessage(`Fluno LSP Start Failed: ${e}`);
        });
    } catch (err) {
        outputChannel.appendLine(`CRITICAL ERROR: ${err}`);
        window.showErrorMessage(`Fluno LSP Critical Error: ${err}`);
    }
}

function deactivate() {
    if (!client) {
        return undefined;
    }
    return client.stop();
}

module.exports = {
    activate,
    deactivate
};
