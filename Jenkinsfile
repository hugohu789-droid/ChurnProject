pipeline {
    agent any
    
    environment {
        // 可根据需要修改 Node 和 Docker 环境
        DEPLOY_SERVER = "test"   // SSH配置中定义的服务器名
    }
    
    tools {
        nodejs 'NodeJS'
    }
    
    stages {
        // -------------------- Step 1: Clone GitHub Code --------------------
        stage('Checkout Code') {
            steps {
                echo "\u001B[34m[Step 1] 🌀 Cloning repository from GitHub...\u001B[0m"
                git branch: 'main',
                    url: 'https://github.com/hugohu789-droid/ChurnProject.git',
                    credentialsId: 'github-pat'
            }
        }
        
        // -------------------- Step 2: Build Frontend --------------------
        // stage('Build Frontend') {
        //     steps {
        //         dir('frontend') {
        //             echo "\u001B[34m[Step 2] 🚀 Installing dependencies and building frontend...\u001B[0m"
        //             // 自动尝试 npm ci，失败则执行 npm install
        //             sh '''
        //             # 1️⃣ 安装依赖
        //             npm ci || npm install
        //             # 4️⃣ 运行构建
        //             npm run build
        //             '''
        //         }
        //     }
        // }
        // -------------------- Step 3: Upload Files to Remote Server --------------------
        stage('Upload Artifacts') {
            steps {
                echo "\u001B[34m[Step 3] 📦 Uploading built files to remote CentOS server...\u001B[0m"
                sshPublisher(publishers: [
                    sshPublisherDesc(
                        configName: "${DEPLOY_SERVER}",
                        transfers: [
                            sshTransfer(
                                sourceFiles: 'frontend/**',
                                removePrefix: 'frontend/',
                                remoteDirectory: "frontend/"
                            ),
                            sshTransfer(
                                sourceFiles: 'backend/**',
                                removePrefix: 'backend',
                                remoteDirectory: "backend"
                            ),
                            sshTransfer(
                                sourceFiles: 'deploy/**',
                                removePrefix: 'deploy',
                                remoteDirectory: "deploy",
                                execCommand: '''
                                    echo "[Step 4] 🐳 Starting Docker containers..."
                                    cd /usr/local/test/deploy
                                    docker compose down
                                    docker compose up -d --build
                                    docker compose ps
                                    echo "[SUCCESS] ✅ All containers started successfully."
                                '''
                            )
                        ],
                        usePromotionTimestamp: false,
                        verbose: true
                    )
                ])
            }
        }
    }
    
    post {
        success {
            echo "\u001B[32m✅ Deployment succeeded! Application is live.\u001B[0m"
        }
        failure {
            echo "\u001B[31m❌ Deployment failed. Please check Jenkins logs.\u001B[0m"
        }
    }
}
