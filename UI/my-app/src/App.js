import React from 'react';
import 'antd/dist/antd.css';
import { Upload, message, Button, Progress } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { Grid, Typography } from '@material-ui/core'

import './App.css'

import axios from "axios"
axios.defaults.withCredentials = true

class App extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      fileList: [],
      uploading: false,
      filseSize: 0,
      baifenbi: 0
    }
  }
  //文件上传改变的时候
  configs = {
    headers: { 'Content-Type': 'multipart/form-data' },
    withCredentials: true,
    onUploadProgress: (progress) => {
      console.log(progress);
      let { loaded } = progress
      let { filseSize } = this.state
      console.log(loaded, filseSize);
      let baifenbi = (loaded / filseSize * 100).toFixed(2)
      this.setState({
        baifenbi
      })
    }
  }
  //点击上传
  handleUpload = () => {
    const { fileList } = this.state;
    const formData = new FormData();
    fileList.forEach(file => {
      formData.append('files[]', file);
    });
    this.setState({
      uploading: true,
    });
    //请求本地服务
    axios.post("http://127.0.0.1:5000/upload", formData, this.configs).then(res => {
      this.setState({
        baifenbi: 100,
        uploading: false,
        fileList: []
      })
    }).finally(log => {
      console.log(log);
    })
  }
  onchange = (info) => {
    if (info.file.status !== 'uploading') {
      this.setState({
        filseSize: info.file.size,
        baifenbi: 0
      })
    }
    if (info.file.status === 'done') {
      message.success(`${info.file.name} file uploaded successfully`);
    } else if (info.file.status === 'error') {
      message.error(`${info.file.name} file upload failed.`);
    }
  }


  render() {
    const { uploading, fileList } = this.state;
    const props = {
      onRemove: file => {
        this.setState(state => {
          const index = state.fileList.indexOf(file);
          const newFileList = state.fileList.slice();
          newFileList.splice(index, 1);
          return {
            fileList: newFileList,
          };
        });
      },
      beforeUpload: file => {
        this.setState(state => ({
          fileList: [...state.fileList, file],
        }));
        return false;
      },
      fileList,
    };
    return (
        <>
          <div class='bg'></div>
          <div class='bg bg2'></div>
          <div class='bg bg3'></div>

          <Grid container justify='center' style={{ marginTop: '50px'}}>
            <Grid item xs={30}>
              <Typography variant='h3' align='center' className='heading'>
                {''}
                Music Transcription System
              </Typography>
            </Grid>
          </Grid>

          <Grid>
            <div style={{ width: "80%", margin: 'auto', padding: 20 }}>
              <h3>{this.state.baifenbi + '%'}</h3>
                <Upload onChange={(e) => { this.onchange(e) }} {...props}>
                  <Button>
                    <UploadOutlined /> Click to Upload MP3 File
                  </Button>
                </Upload>
                <Button
                  type="primary"
                  onClick={this.handleUpload}
                  disabled={fileList.length === 0}
                  loading={uploading}
                  style={{ marginTop: 16 }}
                > {uploading ? 'Uploading' : 'Start Uploading'}
                </Button>
              <Progress style={{ marginTop: 20 }} status={this.state.baifenbi !== 0 ? 'success' : ''} percent={this.state.baifenbi}></Progress>
            </div >
          </Grid>

          <Grid>
            <div className='wrapper'>
              <span className='music-note' id='one'>
                ♫
              </span>
              <span className='music-note' id='two'>
                ♪
              </span>
            </div>
          </Grid>

          <Grid container justify='center'>
            <Grid item>
              <div className='card'>
                <div className='card-text'>
                  <div className='card-dance-name'>
                    <div className='lyrics'>
                      <h3>ALL MY DREAMS FULFILL</h3>
                      <Button variant='contained'color='secondary'>Start Playing Melody</Button>
                    </div>
                    <div className='lyrics'>
                      <h3>FOR MY DARLING ILL LOVEYOU</h3>
                      <script type = "text/javascript" src = "/static/lyrics_script.js"></script>
                      <Button variant='contained'color='secondary'>Start Playing Melody</Button>
                    </div>
                    <div className='lyrics'>
                      <h3>AND I ALWAYS WILL</h3>
                      <Button variant='contained'color='secondary'>Start Playing Melody</Button>
                    </div>
                  </div>
                </div>
                <div className='card-stats'>
                  <div className='stat border'></div>
                  <div className='stat'>
                    <div className='value'>RESULT</div>
                  </div>
                  <div className='stat'></div>
                </div>
              </div>
            </Grid>
          </Grid>

        </>
    )
  }
}

export default App;
