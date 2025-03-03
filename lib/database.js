import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://npdwqviokmkfynkacnpd.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5wZHdxdmlva21rZnlua2FjbnBkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM0NjgwNjAsImV4cCI6MjA0OTA0NDA2MH0.fY6qlzRUISzyAaUHMQcjSO7EftS_LgG1rc2yH13BVqE';

const supabase = createClient(supabaseUrl, supabaseKey);

export default supabase;
