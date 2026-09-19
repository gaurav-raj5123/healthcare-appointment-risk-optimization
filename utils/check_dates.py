import pandas as pd

df = pd.read_csv('cleaned_patient_data.csv')
df['appointment_day'] = pd.to_datetime(df['appointment_day'])
df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])

print('AppointmentDay weekly volume:')
print(df['appointment_day'].dt.isocalendar().week.value_counts().sort_index())

print('\nAppointmentDay dates count per day (first 5 and last 5):')
daily = df['appointment_day'].dt.date.value_counts().sort_index()
print(daily.head(5))
print("...")
print(daily.tail(5))
print(f"\nTotal days with appointments: {len(daily)}")
