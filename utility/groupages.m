function [ age_groups ] = groupages( age_vector )
%UNTİTLED Summary of this function goes here
%   Detailed explanation goes here
age_groups=1*(age_vector<=48)+2*(age_vector>48&age_vector<=66)+3*(age_vector>66);

end

