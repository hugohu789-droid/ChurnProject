import type { TrainingRecord } from './modelTraining'

export const mockTrainingRecords: TrainingRecord[] = [
  {
    id: 1,
    original_filename: 'churn_dataset_v1.csv',
    saved_filename: 'churn_dataset_v1_1234.csv',
    upload_time: '2023-01-01T10:00:00Z',
    file_size: 1024000,
    status: 'trained',
    modelName: 'ChurnModel_v1'
  },
  {
    id: 2,
    original_filename: 'churn_dataset_v2.csv',
    saved_filename: 'churn_dataset_v2_2345.csv',
    upload_time: '2023-01-02T09:00:00Z',
    file_size: 1048576,
    status: 'uploaded'
  },
  {
    id: 3,
    original_filename: 'churn_dataset_v3.csv',
    saved_filename: 'churn_dataset_v3_3456.csv',
    upload_time: '2023-01-03T14:00:00Z',
    file_size: 2048000,
    status: 'trained',
    modelName: 'ChurnModel_v3'
  },
  {
    id: 4,
    original_filename: 'churn_dataset_v4.csv',
    saved_filename: 'churn_dataset_v4_4567.csv',
    upload_time: '2023-01-04T11:00:00Z',
    file_size: 1536000,
    status: 'failed'
  },
  {
    id: 5,
    original_filename: 'churn_dataset_v5.csv',
    saved_filename: 'churn_dataset_v5_5678.csv',
    upload_time: '2023-01-05T16:00:00Z',
    file_size: 1843200,
    status: 'training',
    modelName: 'ChurnModel_v5'
  }
]
