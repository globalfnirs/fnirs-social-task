%{
Function for looking time rejection
Author: Johann Benerradi
------------------------
%}

%{
new_s = looking_time(data, lt_filename, perc_look)

Function to annotate stimulus matrix based on looking time, based on a
provided threshold. Simulus for which the looking time is below threshold
are marked as -2.

INPUTS
------
data: structure
    The .nirs data (with fields data.s, data.CondNames at least).

lt_filename : string
    Path to the looking time autocoder file from the data subject folder.

perc_look: float
    Stimulus for which the looking time is below this threshold (in
    percent) are marked for rejection.


OUTPUTS
-------
new_s : matrix
    The new stiumulus matrix with stimulus marked for rejection based on
    looking time.

file_found : int
    Code indicated whether the autocoder looking time file has been found. 1 for
    found; 0 for not found; -1 for non-matching file.
%}

function [new_s, file_found] = looking_time(data, lt_filename, perc_look)
    
    % Copy stim matrix
    new_s = data.s;
    file_found = 0;

    % Try to open the autocoder looking time file if exists
    try
        lt_data = load(lt_filename, '-mat');
    catch
        warning('No %s file', lt_filename)
        return;
    end

    % Check that number of autocoder entries match number of stims (except C)
    if length(lt_data.mergedCoding.Trial) ~= nnz(data.s(:, 2:end))
        warning('Number of stims in %s not matching with .nirs file', lt_filename)
        file_found = -1;
        return;
    end
    
    % Create looking time table from file
    table_lt = split(lt_data.mergedCoding.Trial, '_');
    table_lt = [table_lt, num2cell(lt_data.mergedCoding.Attended)];
    table_lt = cell2table(table_lt(:, 3:end));
    table_lt.Properties.VariableNames = ["Cond", "Number", "LT"];
    table_lt.Number = str2double(table_lt.Number);

    % Iterate over conditions from the .nirs data
    for cond=data.CondNames
        % Select subtable of that condition
        table_lt_cond = table_lt(strcmp(cond, table_lt.Cond), :);
        % Iterate over all occurences of that condition
        for row = 1:size(table_lt_cond, 1)
            % If the looking time is below threshold mark that stim -2
            if table_lt_cond.LT(row) < (perc_look/100)
                str_cond = string(cond);
                sprintf('Marking stim %s #%d (looking time < %.2f%%)', str_cond, row, perc_look)
                cond_id = find(strcmp(cond, data.CondNames));
                cond_stim_times = find(data.s(:, cond_id) ~= 0);
                cond_stim_time = cond_stim_times(table_lt_cond.Number(row));
                new_s(cond_stim_time, cond_id) = -2;
            end
        end
    end
    file_found = 1;
end
